"""Spike: 验证 anomalib v2.5.1 Patchcore + DINOv2 骨干的层选择与可运行性。

运行（需联网下载 ~90MB DINOv2 权重，CPU 可跑）:
    python scripts/spike_backbone.py

预期输出（形状随 image_size=224 固定）:
    [Patchcore blocks.9]  feature shape: torch.Size([1, 384, 16, 16])  -> OK (4D)
    [Patchcore blocks.11] feature shape: torch.Size([1, 384, 16, 16])  -> OK (4D)
    [Patchcore blocks.9+11] embedding dim: 768 (= 384*2)
    [AnomalyDINO] encoder forward OK, feature shape: ...
    SPIKE RESULT: PASS
出口: 将选定的默认层写回 config/default.yaml 的 layers 与 dino_exp.config.BACKBONE_ALIASES。
"""

import torch
from anomalib.models import Patchcore


def check_patchcore(layers: list[str]) -> None:
    model = Patchcore(
        backbone="vit_small_patch14_dinov2.lvd142m",
        layers=layers,
        pre_trained=True,
    )
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        feats = model.model.feature_extractor(x)
    for name, f in feats.items():
        assert f.dim() == 4, f"层 {name} 输出不是 4D 特征图: {f.shape}"
        print(f"[Patchcore {name}]  feature shape: {tuple(f.shape)}  -> OK (4D)")
    # 验证 generate_embedding + AvgPool 通路（Patchcore forward 的训练分支）
    emb = model.model(x)  # training 模式下返回 (B*H*W, D) embedding
    print(f"[Patchcore {'+'.join(layers)}] embedding dim: {emb.shape[-1]}")


def check_anomaly_dino() -> None:
    from anomalib.models import AnomalyDINO

    model = AnomalyDINO(encoder_name="vit_small_patch14_dinov2")
    x = torch.randn(1, 3, 224, 224)
    model.train()
    with torch.no_grad():
        out = model.model(x)
    print(f"[AnomalyDINO] encoder forward OK, output type: {type(out).__name__}")


if __name__ == "__main__":
    check_patchcore(["blocks.9"])
    check_patchcore(["blocks.11"])
    check_patchcore(["blocks.9", "blocks.11"])
    check_anomaly_dino()
    print("SPIKE RESULT: PASS")
