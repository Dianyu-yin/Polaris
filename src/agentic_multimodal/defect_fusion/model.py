"""Learnable defect-effect branch fused with the packaged hybrid CNN.

The CNN skeleton is the existing HybridConvNeXtDenseNet instance. This module
wraps it, exposes its existing image features, and adds trainable parameters for
survey-aligned defect identity and Qwen severity 1-3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from agentic_multimodal.defect_detection import DEFAULT_DEFECTS


IMAGE_FEATURE_DIM = 768 + 1024
SURVEY_SCHEMA_VERSION = "survey_defect_v2"


@dataclass(frozen=True)
class DefectConfig:
    defect_names: tuple[str, ...]

    @property
    def defect_to_id(self) -> dict[str, int]:
        return {name: idx + 1 for idx, name in enumerate(self.defect_names)}


def build_default_defect_config() -> DefectConfig:
    """这个函数把问卷中的四类缺陷固定为 fusion 模型的 canonical taxonomy。"""

    defect_names = tuple(name for name, _description in DEFAULT_DEFECTS)
    return DefectConfig(defect_names=defect_names)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _unfreeze_last_children(module: nn.Module, n_children: int) -> None:
    if n_children <= 0:
        return
    children = list(module.children())
    for child in children[-n_children:]:
        _set_requires_grad(child, True)


def configure_trainable_cnn_layers(
    cnn: nn.Module,
    trainable_convnext_blocks: int = 1,
    trainable_densenet_blocks: int = 2,
    train_heads: bool = True,
) -> dict[str, int]:
    """Freeze the CNN backbone, then unfreeze late layers and existing heads."""

    _set_requires_grad(cnn, False)
    _unfreeze_last_children(cnn.convnext_features, trainable_convnext_blocks)
    _unfreeze_last_children(cnn.densenet_features, trainable_densenet_blocks)
    if train_heads:
        _set_requires_grad(cnn.cls_head, True)
        _set_requires_grad(cnn.reg_head, True)

    return {
        "total_parameters": sum(parameter.numel() for parameter in cnn.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in cnn.parameters() if parameter.requires_grad),
        "trainable_convnext_blocks": trainable_convnext_blocks,
        "trainable_densenet_blocks": trainable_densenet_blocks,
        "train_heads": int(train_heads),
    }


def freeze_cnn_completely(cnn: nn.Module) -> dict[str, int]:
    """这个函数冻结整个 CNN 并切换到 eval，确保 fusion 训练不更新参数或 BN 统计。"""

    _set_requires_grad(cnn, False)
    cnn.eval()
    return {
        "total_parameters": sum(parameter.numel() for parameter in cnn.parameters()),
        "trainable_parameters": 0,
        "trainable_convnext_blocks": 0,
        "trainable_densenet_blocks": 0,
        "train_heads": 0,
    }


class DefectEffectEncoder(nn.Module):
    """这个编码层把问卷对齐的 defect type 与 1-3 severity 编成 token 特征。"""

    def __init__(
        self,
        defect_config: DefectConfig,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.defect_config = defect_config
        self.defect_to_id = defect_config.defect_to_id
        self.output_dim = output_dim

        n_defects = len(defect_config.defect_names)
        self.defect_embedding = nn.Embedding(n_defects + 1, embedding_dim, padding_idx=0)
        self.llm_severity_embedding = nn.Embedding(4, embedding_dim, padding_idx=0)

        self.defect_effect = nn.Embedding(n_defects + 1, 1, padding_idx=0)
        self.llm_severity_effect = nn.Embedding(4, 1, padding_idx=0)

        token_input_dim = embedding_dim * 2 + 3
        self.token_mlp = nn.Sequential(
            nn.Linear(token_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )
        self.empty_defect_feature = nn.Parameter(torch.zeros(output_dim))

    def encode_tokens(self, defect_ids: torch.Tensor, llm_severity_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """这个函数保留每个 defect 的 token，供 Transformer 在融合阶段使用。"""

        if defect_ids.ndim != 2 or llm_severity_ids.ndim != 2:
            raise ValueError("defect_ids and llm_severity_ids must both be [batch, n_defects] tensors")
        if defect_ids.shape != llm_severity_ids.shape:
            raise ValueError("defect_ids and llm_severity_ids must have the same shape")

        defect_ids = defect_ids.clamp(min=0, max=self.defect_embedding.num_embeddings - 1)
        llm_severity_ids = llm_severity_ids.clamp(min=0, max=3)

        defect_emb = self.defect_embedding(defect_ids)
        llm_emb = self.llm_severity_embedding(llm_severity_ids)

        defect_effect = self.defect_effect(defect_ids)
        llm_effect = self.llm_severity_effect(llm_severity_ids)
        llm_numeric = llm_severity_ids.float().unsqueeze(-1) / 3.0

        token_input = torch.cat(
            [
                defect_emb,
                llm_emb,
                defect_effect,
                llm_effect,
                llm_numeric,
            ],
            dim=-1,
        )
        token_features = self.token_mlp(token_input)
        mask = defect_ids.ne(0)
        return token_features, mask

    def forward(self, defect_ids: torch.Tensor, llm_severity_ids: torch.Tensor) -> torch.Tensor:
        """这个函数把多个 defect token 池化成一个固定长度 defect feature。"""

        token_features, token_mask = self.encode_tokens(defect_ids, llm_severity_ids)
        mask = token_mask.unsqueeze(-1)
        masked_features = token_features * mask
        counts = mask.sum(dim=1)
        pooled = masked_features.sum(dim=1) / counts.clamp_min(1)
        empty_rows = counts.squeeze(-1).eq(0)
        if empty_rows.any():
            pooled = pooled.clone()
            pooled[empty_rows] = self.empty_defect_feature
        return pooled


class DefectFusionPCEModel(nn.Module):
    """这个模型使用 MLP 做 late fusion，用于兼容已有 checkpoint。"""

    def __init__(
        self,
        cnn: nn.Module,
        defect_encoder: DefectEffectEncoder,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.2,
        residual_pce: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = cnn
        self.defect_encoder = defect_encoder
        self.residual_pce = residual_pce

        fusion_input_dim = IMAGE_FEATURE_DIM + 1 + 3 + defect_encoder.output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        convnext_feat = self.cnn.convnext_pool(self.cnn.convnext_features(images)).flatten(1)
        densenet_feat = self.cnn.densenet_pool(self.cnn.densenet_features(images)).flatten(1)
        return torch.cat([convnext_feat, densenet_feat], dim=1)

    def set_frozen_backbone_eval(self) -> None:
        """Keep fully frozen feature children in eval mode during training."""

        if not any(parameter.requires_grad for parameter in self.cnn.parameters()):
            self.cnn.eval()
            return
        for feature_module in (self.cnn.convnext_features, self.cnn.densenet_features):
            for child in feature_module.children():
                if not any(parameter.requires_grad for parameter in child.parameters()):
                    child.eval()

    def forward(
        self,
        images: torch.Tensor,
        defect_ids: torch.Tensor,
        llm_severity_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_features = self.extract_image_features(images)
        logits = self.cnn.cls_head(image_features)
        class_probabilities = torch.softmax(logits, dim=1)
        cnn_pce = self.cnn.reg_head(image_features).squeeze(1)
        defect_features = self.defect_encoder(defect_ids, llm_severity_ids)

        fusion_input = torch.cat(
            [
                image_features,
                cnn_pce.unsqueeze(1),
                class_probabilities,
                defect_features,
            ],
            dim=1,
        )
        pce_delta = self.fusion_head(fusion_input).squeeze(1)
        final_pce = cnn_pce + pce_delta if self.residual_pce else pce_delta
        return {
            "final_pce": final_pce,
            "cnn_pce": cnn_pce,
            "pce_delta": pce_delta,
            "logits": logits,
            "class_probabilities": class_probabilities,
            "defect_features": defect_features,
        }

    def export_config(self) -> dict[str, Any]:
        return {
            "architecture": "mlp",
            "schema_version": SURVEY_SCHEMA_VERSION,
            "defect_names": list(self.defect_encoder.defect_config.defect_names),
            "residual_pce": self.residual_pce,
        }


class CnnOnlyHybridPCEModel(nn.Module):
    """这个模型只使用原始 ConvNeXt 和 DenseNet 双分支融合特征，不接入 Qwen defect token。"""

    def __init__(self, cnn: nn.Module) -> None:
        super().__init__()
        self.cnn = cnn

    def set_frozen_backbone_eval(self) -> None:
        """这个函数让已经冻结的 CNN 子模块保持 eval 状态，避免 BatchNorm/Dropout 漂移。"""

        if not any(parameter.requires_grad for parameter in self.cnn.parameters()):
            self.cnn.eval()
            return
        for feature_module in (self.cnn.convnext_features, self.cnn.densenet_features):
            for child in feature_module.children():
                if not any(parameter.requires_grad for parameter in child.parameters()):
                    child.eval()

    def forward(
        self,
        images: torch.Tensor,
        defect_ids: torch.Tensor | None = None,
        llm_severity_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """这个函数只用 CNN 原始双 backbone 输出分类 logits 和 PCE 回归值。"""

        logits, cnn_pce = self.cnn(images)
        class_probabilities = torch.softmax(logits, dim=1)
        pce_delta = torch.zeros_like(cnn_pce)
        return {
            "final_pce": cnn_pce,
            "cnn_pce": cnn_pce,
            "pce_delta": pce_delta,
            "logits": logits,
            "cnn_logits": logits,
            "class_probabilities": class_probabilities,
            "defect_features": torch.empty((images.shape[0], 0), device=images.device, dtype=cnn_pce.dtype),
        }

    def export_config(self) -> dict[str, Any]:
        return {
            "architecture": "cnn_only",
            "residual_pce": False,
        }


class TransformerDefectFusionPCEModel(nn.Module):
    """这个模型让 CNN 图像 token 和 Qwen defect token 做 attention，再预测 CNN PCE 的残差修正。"""

    def __init__(
        self,
        cnn: nn.Module,
        defect_encoder: DefectEffectEncoder,
        transformer_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 1,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.1,
        residual_pce: bool = True,
        use_null_semantic_token: bool = False,
    ) -> None:
        super().__init__()
        if transformer_dim % transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads")

        self.cnn = cnn
        self.defect_encoder = defect_encoder
        self.transformer_dim = transformer_dim
        self.residual_pce = residual_pce
        self.use_null_semantic_token = use_null_semantic_token

        self.convnext_projection = nn.Linear(768, transformer_dim)
        self.densenet_projection = nn.Linear(1024, transformer_dim)
        self.defect_projection = nn.Linear(defect_encoder.output_dim, transformer_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.null_semantic_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.token_type_embedding = nn.Embedding(4, transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(transformer_dim)

        self.cls_head = nn.Linear(transformer_dim, 3)
        regression_input_dim = transformer_dim + 4 if residual_pce else transformer_dim
        self.reg_head = nn.Sequential(
            nn.Linear(regression_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )
        if residual_pce:
            nn.init.zeros_(self.reg_head[-1].weight)
            nn.init.zeros_(self.reg_head[-1].bias)

    def extract_branch_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """这个函数从原 CNN 骨架中取出 ConvNeXt 和 DenseNet 两个分支的视觉特征。"""

        convnext_feat = self.cnn.convnext_pool(self.cnn.convnext_features(images)).flatten(1)
        densenet_feat = self.cnn.densenet_pool(self.cnn.densenet_features(images)).flatten(1)
        return convnext_feat, densenet_feat

    def set_frozen_backbone_eval(self) -> None:
        """这个函数让已经冻结的 CNN 子模块保持 eval 状态，避免 BatchNorm/Dropout 漂移。"""

        if not any(parameter.requires_grad for parameter in self.cnn.parameters()):
            self.cnn.eval()
            return
        for feature_module in (self.cnn.convnext_features, self.cnn.densenet_features):
            for child in feature_module.children():
                if not any(parameter.requires_grad for parameter in child.parameters()):
                    child.eval()

    def forward(
        self,
        images: torch.Tensor,
        defect_ids: torch.Tensor,
        llm_severity_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        convnext_feat, densenet_feat = self.extract_branch_features(images)
        image_features = torch.cat([convnext_feat, densenet_feat], dim=1)

        cnn_logits = self.cnn.cls_head(image_features)
        cnn_pce = self.cnn.reg_head(image_features).squeeze(1)
        cnn_class_probabilities = torch.softmax(cnn_logits, dim=1)

        defect_tokens, defect_mask = self.defect_encoder.encode_tokens(defect_ids, llm_severity_ids)
        batch_size = images.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        convnext_token = self.convnext_projection(convnext_feat).unsqueeze(1)
        densenet_token = self.densenet_projection(densenet_feat).unsqueeze(1)
        projected_defect_tokens = self.defect_projection(defect_tokens)
        if self.use_null_semantic_token:
            empty_rows = ~defect_mask.any(dim=1)
            if empty_rows.any():
                projected_defect_tokens = projected_defect_tokens.clone()
                defect_mask = defect_mask.clone()
                # 这个 cast 让可学习的 FP32 null token 在 AMP 下匹配投影 token 的计算 dtype/device。
                null_semantic_token = self.null_semantic_token[0, 0, :].to(
                    dtype=projected_defect_tokens.dtype,
                    device=projected_defect_tokens.device,
                )
                projected_defect_tokens[empty_rows, 0, :] = null_semantic_token
                defect_mask[empty_rows, 0] = True

        tokens = torch.cat([cls_token, convnext_token, densenet_token, projected_defect_tokens], dim=1)
        token_type_ids = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=images.device),
                torch.ones((batch_size, 1), dtype=torch.long, device=images.device),
                torch.full((batch_size, 1), 2, dtype=torch.long, device=images.device),
                torch.full(defect_ids.shape, 3, dtype=torch.long, device=images.device),
            ],
            dim=1,
        )
        tokens = tokens + self.token_type_embedding(token_type_ids)

        valid_mask = torch.cat(
            [
                torch.ones((batch_size, 3), dtype=torch.bool, device=images.device),
                defect_mask.to(images.device),
            ],
            dim=1,
        )
        fused_tokens = self.transformer(tokens, src_key_padding_mask=~valid_mask)
        cls_state = self.norm(fused_tokens[:, 0])

        logits = self.cls_head(cls_state)
        class_probabilities = torch.softmax(logits, dim=1)
        regression_input = (
            torch.cat([cls_state, cnn_pce.unsqueeze(1), cnn_class_probabilities], dim=1)
            if self.residual_pce
            else cls_state
        )
        regression_output = self.reg_head(regression_input).squeeze(1)
        final_pce = cnn_pce + regression_output if self.residual_pce else regression_output
        return {
            "final_pce": final_pce,
            "cnn_pce": cnn_pce,
            "pce_delta": final_pce - cnn_pce,
            "logits": logits,
            "cnn_logits": cnn_logits,
            "class_probabilities": class_probabilities,
            "cnn_class_probabilities": cnn_class_probabilities,
            "defect_features": cls_state,
        }

    def export_config(self) -> dict[str, Any]:
        return {
            "architecture": "transformer",
            "schema_version": SURVEY_SCHEMA_VERSION,
            "defect_names": list(self.defect_encoder.defect_config.defect_names),
            "transformer_dim": self.transformer_dim,
            "residual_pce": self.residual_pce,
            "use_null_semantic_token": self.use_null_semantic_token,
        }
