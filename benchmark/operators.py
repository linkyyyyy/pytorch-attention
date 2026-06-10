"""
operators.py — ONNX operator registry and graph export for energy benchmarking.

Each registry entry maps to (build_fn, shape_profiles, dtype, cluster).
Exports minimal single-operator graphs at ONNX opset 20 (torch.onnx.export falls back to 19).
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

PREFERRED_OPSET = 20
FALLBACK_OPSET = 19

ONNX_GRAPHS_DIR = Path(__file__).parent / "onnx_graphs"
RNG = np.random.default_rng(0)


@dataclass(frozen=True)
class OperatorEntry:
    name: str
    build_fn: Callable[[Path, dict[str, Any], int], dict[str, Any]]
    shape_profiles: list[dict[str, Any]]
    dtype: str
    cluster: str


@dataclass(frozen=True)
class SdpaBlockConfig:
    """Single source of shape truth for one SDPA attention-block corner."""

    block_id: str
    shape_class: str
    N: int
    D: int
    num_heads: int
    head_dim: int
    mlp_ratio: int = 4

    @property
    def D_ff(self) -> int:
        return self.D * self.mlp_ratio


# DSE corners — all isolated-op shapes for SDPA block ops derive from these only.
SDPA_BLOCK_CONFIGS: tuple[SdpaBlockConfig, ...] = (
    SdpaBlockConfig(
        block_id="sdpa_small",
        shape_class="small",
        N=49,
        D=768,
        num_heads=12,
        head_dim=64,
    ),
    SdpaBlockConfig(
        block_id="sdpa_avg",
        shape_class="avg",
        N=197,
        D=768,
        num_heads=12,
        head_dim=64,
    ),
    SdpaBlockConfig(
        block_id="sdpa_large",
        shape_class="large",
        N=3136,
        D=768,
        num_heads=12,
        head_dim=64,
    ),
)

SDPA_BLOCK_CONFIG_BY_ID: dict[str, SdpaBlockConfig] = {
    cfg.block_id: cfg for cfg in SDPA_BLOCK_CONFIGS
}


def _block_context(cfg: SdpaBlockConfig, *, fusion_member: bool) -> dict[str, Any]:
    return {
        "block_id": cfg.block_id,
        "shape_class": cfg.shape_class,
        "tier": "isolated",
        "fusion_member": fusion_member,
    }


def _gemm_profile(
    cfg: SdpaBlockConfig,
    label_suffix: str,
    m: int,
    k: int,
    n: int,
    *,
    fusion_member: bool,
) -> dict[str, Any]:
    return {
        "label": f"{cfg.block_id}_{label_suffix}",
        "input_shape": f"{m}x{k}@{k}x{n}",
        "tensors": {"m": m, "k": k, "n": n},
        **_block_context(cfg, fusion_member=fusion_member),
    }


def _matmul_profile(
    cfg: SdpaBlockConfig,
    label_suffix: str,
    a_shape: list[int],
    b_shape: list[int],
    *,
    fusion_member: bool,
) -> dict[str, Any]:
    a_str = "x".join(str(d) for d in a_shape)
    b_str = "x".join(str(d) for d in b_shape)
    return {
        "label": f"{cfg.block_id}_{label_suffix}",
        "input_shape": f"{a_str}@{b_str}",
        "tensors": {"a_shape": a_shape, "b_shape": b_shape},
        **_block_context(cfg, fusion_member=fusion_member),
    }


def _tensor_profile(
    cfg: SdpaBlockConfig,
    label_suffix: str,
    shape: list[int],
    *,
    fusion_member: bool,
    axis: int | None = None,
) -> dict[str, Any]:
    tensors: dict[str, Any] = {"shape": shape}
    if axis is not None:
        tensors["axis"] = axis
    profile: dict[str, Any] = {
        "label": f"{cfg.block_id}_{label_suffix}",
        "input_shape": "x".join(str(d) for d in shape),
        "tensors": tensors,
        **_block_context(cfg, fusion_member=fusion_member),
    }
    return profile


def derive_isolated_profiles_for_block(cfg: SdpaBlockConfig) -> dict[str, list[dict[str, Any]]]:
    """
    Derive every isolated-operator shape profile for one SDPA block corner.
    Summing runs that share cfg.block_id must reconstruct the ops inside that block.

    fusion_member=True marks attention-core ops included in Tier-1-vs-Tier-2 fusion sums.
    fusion_member=False marks structural context ops (LN, FFN, residual) still measured
    at the same (N, D) but excluded from that fusion comparison.
    """
    H, N, D, hd = cfg.num_heads, cfg.N, cfg.D, cfg.head_dim
    D_ff = cfg.D_ff
    core = True
    ctx = False
    return {
        "qkv_proj_gemm": [
            _gemm_profile(cfg, "q_proj", N, D, D, fusion_member=core),
            _gemm_profile(cfg, "k_proj", N, D, D, fusion_member=core),
            _gemm_profile(cfg, "v_proj", N, D, D, fusion_member=core),
        ],
        "out_proj_gemm": [
            _gemm_profile(cfg, "out_proj", N, D, D, fusion_member=core),
        ],
        "attn_score_matmul": [
            _matmul_profile(
                cfg,
                "attn_qkt",
                [H, N, hd],
                [H, hd, N],
                fusion_member=core,
            ),
        ],
        "softmax": [
            _tensor_profile(cfg, "attn_softmax", [H, N, N], fusion_member=core, axis=-1),
        ],
        "attn_value_matmul": [
            _matmul_profile(
                cfg,
                "attn_av",
                [H, N, N],
                [H, N, hd],
                fusion_member=core,
            ),
        ],
        "layer_norm": [
            _tensor_profile(cfg, "layer_norm", [N, D], fusion_member=ctx),
        ],
        "ffn_gemm": [
            _gemm_profile(cfg, "ffn_expand", N, D, D_ff, fusion_member=ctx),
            _gemm_profile(cfg, "ffn_contract", N, D_ff, D, fusion_member=ctx),
        ],
        "gelu": [
            _tensor_profile(cfg, "ffn_gelu", [N, D_ff], fusion_member=ctx),
        ],
        "residual_add": [
            _tensor_profile(cfg, "residual_add", [N, D], fusion_member=ctx),
        ],
    }


def _merge_sdpa_profiles(
    op_name: str,
    *,
    extra_profiles: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for cfg in SDPA_BLOCK_CONFIGS:
        profiles.extend(derive_isolated_profiles_for_block(cfg)[op_name])
    profiles.extend(extra_profiles)
    return profiles


def _sdpa_dims_from_profile(profile: dict[str, Any]) -> tuple[int, int, int, int]:
    """Read N, D, num_heads, head_dim from a fused-block shape profile."""
    t = profile["tensors"]
    return int(t["N"]), int(t["D"]), int(t["num_heads"]), int(t["head_dim"])


def _fused_block_profile(cfg: SdpaBlockConfig) -> dict[str, Any]:
    H, N, D, hd = cfg.num_heads, cfg.N, cfg.D, cfg.head_dim
    return {
        "label": cfg.block_id,
        "input_shape": f"{N}x{D}",
        "tensors": {
            "N": N,
            "D": D,
            "num_heads": H,
            "head_dim": hd,
        },
        "block_id": cfg.block_id,
        "shape_class": cfg.shape_class,
        "tier": "fused_block",
        "fusion_member": True,
    }


def _fused_block_profiles() -> list[dict[str, Any]]:
    """One fused attention-core graph per SDPA corner (shape_index 0/1/2)."""
    return [_fused_block_profile(cfg) for cfg in SDPA_BLOCK_CONFIGS]


OPSET = PREFERRED_OPSET


def _onnx_path(name: str, shape_index: int) -> Path:
    if shape_index == 0:
        return ONNX_GRAPHS_DIR / f"{name}.onnx"
    return ONNX_GRAPHS_DIR / f"{name}_s{shape_index}.onnx"


def export_torch_module(
    module,
    path: Path,
    dummy_inputs: tuple,
    input_names: list[str],
    output_names: list[str],
    opset: int = OPSET,
) -> int:
    """Export via PyTorch; fall back to opset 19 if opset 20 export fails."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    export_input = dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0]
    with torch.no_grad():
        try:
            torch.onnx.export(
                module,
                export_input,
                str(path),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                dynamo=False,
            )
            return opset
        except Exception as exc:
            if opset == FALLBACK_OPSET:
                raise
            warnings.warn(
                f"torch.onnx.export failed at opset {opset} for {path.name}; "
                f"retrying opset {FALLBACK_OPSET}: {exc}",
                stacklevel=2,
            )
            torch.onnx.export(
                module,
                export_input,
                str(path),
                input_names=input_names,
                output_names=output_names,
                opset_version=FALLBACK_OPSET,
                dynamo=False,
            )
            return FALLBACK_OPSET


def _batched_matmul_output_shape(a_shape: list[int], b_shape: list[int]) -> list[int]:
    return a_shape[:-2] + [a_shape[-2], b_shape[-1]]


def build_onnx_matmul_graph(
    path: Path,
    a_shape: list[int],
    b_shape: list[int],
    opset: int,
    input_a: str = "A",
    input_b: str = "B",
    output_name: str = "Y",
) -> dict[str, Any]:
    import onnx
    from onnx import TensorProto, helper

    path.parent.mkdir(parents=True, exist_ok=True)
    y_shape = _batched_matmul_output_shape(a_shape, b_shape)
    a_info = helper.make_tensor_value_info(input_a, TensorProto.FLOAT, a_shape)
    b_info = helper.make_tensor_value_info(input_b, TensorProto.FLOAT, b_shape)
    y_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, y_shape)
    node = helper.make_node("MatMul", [input_a, input_b], [output_name])
    graph = helper.make_graph([node], "matmul", [a_info, b_info], [y_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return {
        "feeds": {
            input_a: RNG.standard_normal(a_shape, dtype=np.float32),
            input_b: RNG.standard_normal(b_shape, dtype=np.float32),
        },
        "output_names": [output_name],
    }


def build_onnx_softmax_graph(
    path: Path,
    shape: list[int],
    axis: int,
    opset: int,
) -> dict[str, Any]:
    import onnx
    from onnx import TensorProto, helper

    path.parent.mkdir(parents=True, exist_ok=True)
    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    node = helper.make_node("Softmax", ["input"], ["output"], axis=axis)
    graph = helper.make_graph([node], "softmax", [x_info], [y_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return {
        "feeds": {"input": RNG.standard_normal(shape, dtype=np.float32)},
        "output_names": ["output"],
    }


def build_onnx_residual_add_graph(path: Path, shape: list[int], opset: int) -> dict[str, Any]:
    import onnx
    from onnx import TensorProto, helper

    path.parent.mkdir(parents=True, exist_ok=True)
    a_info = helper.make_tensor_value_info("A", TensorProto.FLOAT, shape)
    b_info = helper.make_tensor_value_info("B", TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)
    node = helper.make_node("Add", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "residual_add", [a_info, b_info], [y_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return {
        "feeds": {
            "A": RNG.standard_normal(shape, dtype=np.float32),
            "B": RNG.standard_normal(shape, dtype=np.float32),
        },
        "output_names": ["Y"],
    }


def build_dispatch_baseline_graph(path: Path, opset: int) -> dict[str, Any]:
    """
    Non-elidable dispatch baseline: Add(runtime_input, constant).
    Must NOT use two initializers — ORT_ENABLE_ALL would constant-fold that away.
    """
    import onnx
    from onnx import TensorProto, helper

    path.parent.mkdir(parents=True, exist_ok=True)
    shape = [1]
    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    const = helper.make_tensor("const_one", TensorProto.FLOAT, shape, [1.0])
    node = helper.make_node("Add", ["input", "const_one"], ["output"])
    graph = helper.make_graph([node], "dispatch_baseline", [x_info], [y_info], [const])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return {
        "feeds": {"input": np.array([0.5], dtype=np.float32)},
        "output_names": ["output"],
    }


def build_onnx_gelu_graph(path: Path, shape: list[int], opset: int) -> dict[str, Any]:
    """Direct single-node Gelu graph (bypasses PyTorch export decomposition)."""
    import onnx
    from onnx import TensorProto, helper

    path.parent.mkdir(parents=True, exist_ok=True)
    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    node = helper.make_node("Gelu", ["input"], ["output"])
    graph = helper.make_graph([node], "gelu", [x_info], [y_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return {
        "feeds": {"input": RNG.standard_normal(shape, dtype=np.float32)},
        "output_names": ["output"],
    }


# ---------------------------------------------------------------------------
# Per-operator build functions
# ---------------------------------------------------------------------------


def _build_patch_embed(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    class PatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    x = torch.randn(1, 3, 224, 224)
    export_torch_module(PatchEmbed().eval(), path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _build_downsample_conv(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    class Downsample(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    x = torch.randn(1, 768, 56, 56)
    export_torch_module(Downsample().eval(), path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _linear_gemm_dims(profile: dict[str, Any]) -> tuple[int, int, int]:
    t = profile["tensors"]
    return int(t["m"]), int(t["k"]), int(t["n"])


def _build_ffn_gemm(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    return _build_linear_gemm(path, profile, opset)


def _build_gelu(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    shape = profile["tensors"]["shape"]
    return build_onnx_gelu_graph(path, shape, opset)


def _build_layer_norm(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    shape = profile["tensors"]["shape"]
    n_tokens, embed_dim = shape
    model = nn.LayerNorm(embed_dim).eval()
    x = torch.randn(n_tokens, embed_dim)
    export_torch_module(model, path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _build_group_norm(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    model = nn.GroupNorm(num_groups=1, num_channels=768).eval()
    x = torch.randn(1, 768, 14, 14)
    export_torch_module(model, path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _build_batch_norm(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    model = nn.BatchNorm2d(256).eval()
    x = torch.randn(1, 256, 56, 56)
    export_torch_module(model, path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _build_residual_add(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    shape = profile["tensors"]["shape"]
    return build_onnx_residual_add_graph(path, shape, opset)


def _build_linear_gemm(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    m, k, n = _linear_gemm_dims(profile)
    model = nn.Linear(k, n, bias=True).eval()
    x = torch.randn(m, k)
    export_torch_module(model, path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _build_qkv_proj(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    return _build_linear_gemm(path, profile, opset)


def _build_out_proj(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    return _build_linear_gemm(path, profile, opset)


def _build_attn_score_matmul(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    t = profile["tensors"]
    return build_onnx_matmul_graph(path, t["a_shape"], t["b_shape"], opset)


def _build_xcit_cov_matmul(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    t = profile["tensors"]
    return build_onnx_matmul_graph(path, t["a_shape"], t["b_shape"], opset)


def _build_attn_value_matmul(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    t = profile["tensors"]
    return build_onnx_matmul_graph(path, t["a_shape"], t["b_shape"], opset)


def _build_softmax(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    t = profile["tensors"]
    return build_onnx_softmax_graph(path, t["shape"], t["axis"], opset)


def _build_attn_block_fused(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    """
    Export unfused ONNX for the full SDPA attention core as one sequential graph.
    Shapes come from profile tensors (SdpaBlockConfig) — same source as isolated ops.
    """
    import torch
    import torch.nn as nn

    n_tokens, embed_dim, num_heads, head_dim = _sdpa_dims_from_profile(profile)
    assert embed_dim == num_heads * head_dim

    class FusedAttentionCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.scale = head_dim ** -0.5
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [N, D]
            q = self.q_proj(x).view(n_tokens, self.num_heads, self.head_dim).transpose(0, 1)
            k = self.k_proj(x).view(n_tokens, self.num_heads, self.head_dim).transpose(0, 1)
            v = self.v_proj(x).view(n_tokens, self.num_heads, self.head_dim).transpose(0, 1)
            # [H, N, head_dim] per projection

            # TODO(tower): try ORT com.microsoft.MultiHeadAttention / Attention fused contrib op
            # and verify DirectML + Vitis AI support — replace the unfused MatMul+Softmax+MatMul
            # chain below once EP compatibility is confirmed on the HX 370 tower.
            attn = (q @ k.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v

            out = out.transpose(0, 1).reshape(n_tokens, embed_dim)
            return self.out_proj(out)

    x = torch.randn(n_tokens, embed_dim)
    model = FusedAttentionCore().eval()
    export_torch_module(model, path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _build_sra_conv(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    class SRAConv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(768, 768, kernel_size=8, stride=8, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    x = torch.randn(1, 768, 56, 56)
    export_torch_module(SRAConv().eval(), path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _build_avg_pool(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    class AvgPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.pool(x)

    x = torch.randn(1, 768, 14, 14)
    export_torch_module(AvgPool().eval(), path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _build_depthwise_conv(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    class DepthwiseConv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(768, 768, kernel_size=3, padding=1, groups=768, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    x = torch.randn(1, 768, 14, 14)
    export_torch_module(DepthwiseConv().eval(), path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# XCiT cross-covariance matmul — not derived from SDPA block configs.
_XCIT_COV_PROFILE = {
    "label": "xcit_cross_cov",
    "input_shape": "12x64x197@12x197x64",
    "tensors": {"a_shape": [12, 64, 197], "b_shape": [12, 197, 64]},
}

_XCIT_SOFTMAX_PROFILE = {
    "label": "xcit_channel_matrix",
    "input_shape": "12x64x64",
    "tensors": {"shape": [12, 64, 64], "axis": -1},
}

OPERATOR_REGISTRY: dict[str, OperatorEntry] = {
    "patch_embed_conv2d": OperatorEntry(
        name="patch_embed_conv2d",
        build_fn=_build_patch_embed,
        shape_profiles=[{
            "label": "vit_stem",
            "input_shape": "1x3x224x224",
            "tensors": {},
        }],
        dtype="float32",
        cluster="A",
    ),
    "downsample_conv2d": OperatorEntry(
        name="downsample_conv2d",
        build_fn=_build_downsample_conv,
        shape_profiles=[{
            "label": "pvt_ef_downsample",
            "input_shape": "1x768x56x56",
            "tensors": {},
        }],
        dtype="float32",
        cluster="A",
    ),
    "ffn_gemm": OperatorEntry(
        name="ffn_gemm",
        build_fn=_build_ffn_gemm,
        shape_profiles=_merge_sdpa_profiles("ffn_gemm"),
        dtype="float32",
        cluster="A",
    ),
    "gelu": OperatorEntry(
        name="gelu",
        build_fn=_build_gelu,
        shape_profiles=_merge_sdpa_profiles("gelu"),
        dtype="float32",
        cluster="A",
    ),
    "layer_norm": OperatorEntry(
        name="layer_norm",
        build_fn=_build_layer_norm,
        shape_profiles=_merge_sdpa_profiles("layer_norm"),
        dtype="float32",
        cluster="A",
    ),
    "group_norm": OperatorEntry(
        name="group_norm",
        build_fn=_build_group_norm,
        shape_profiles=[{
            "label": "poolformer_gn1",
            "input_shape": "1x768x14x14,groups=1",
            "tensors": {},
        }],
        dtype="float32",
        cluster="A",
    ),
    "batch_norm": OperatorEntry(
        name="batch_norm",
        build_fn=_build_batch_norm,
        shape_profiles=[{
            "label": "ef_batchnorm",
            "input_shape": "1x256x56x56",
            "tensors": {},
        }],
        dtype="float32",
        cluster="A",
    ),
    "residual_add": OperatorEntry(
        name="residual_add",
        build_fn=_build_residual_add,
        shape_profiles=_merge_sdpa_profiles("residual_add"),
        dtype="float32",
        cluster="A",
    ),
    "qkv_proj_gemm": OperatorEntry(
        name="qkv_proj_gemm",
        build_fn=_build_qkv_proj,
        shape_profiles=_merge_sdpa_profiles("qkv_proj_gemm"),
        dtype="float32",
        cluster="B1",
    ),
    "out_proj_gemm": OperatorEntry(
        name="out_proj_gemm",
        build_fn=_build_out_proj,
        shape_profiles=_merge_sdpa_profiles("out_proj_gemm"),
        dtype="float32",
        cluster="B1",
    ),
    "attn_score_matmul": OperatorEntry(
        name="attn_score_matmul",
        build_fn=_build_attn_score_matmul,
        shape_profiles=_merge_sdpa_profiles("attn_score_matmul"),
        dtype="float32",
        cluster="B1",
    ),
    "xcit_cov_matmul": OperatorEntry(
        name="xcit_cov_matmul",
        build_fn=_build_xcit_cov_matmul,
        shape_profiles=[_XCIT_COV_PROFILE],
        dtype="float32",
        cluster="B1",
    ),
    "softmax": OperatorEntry(
        name="softmax",
        build_fn=_build_softmax,
        shape_profiles=_merge_sdpa_profiles(
            "softmax",
            extra_profiles=[_XCIT_SOFTMAX_PROFILE],
        ),
        dtype="float32",
        cluster="B1",
    ),
    "attn_value_matmul": OperatorEntry(
        name="attn_value_matmul",
        build_fn=_build_attn_value_matmul,
        shape_profiles=_merge_sdpa_profiles("attn_value_matmul"),
        dtype="float32",
        cluster="B1",
    ),
    "attn_block_fused": OperatorEntry(
        name="attn_block_fused",
        build_fn=_build_attn_block_fused,
        shape_profiles=_fused_block_profiles(),
        dtype="float32",
        cluster="B1",
    ),
    "sra_conv2d": OperatorEntry(
        name="sra_conv2d",
        build_fn=_build_sra_conv,
        shape_profiles=[{
            "label": "pvt_sra",
            "input_shape": "1x768x56x56",
            "tensors": {},
        }],
        dtype="float32",
        cluster="B1",
    ),
    "avg_pool_token_mixer": OperatorEntry(
        name="avg_pool_token_mixer",
        build_fn=_build_avg_pool,
        shape_profiles=[{
            "label": "poolformer_mixer",
            "input_shape": "1x768x14x14",
            "tensors": {},
        }],
        dtype="float32",
        cluster="B2",
    ),
    "depthwise_conv2d": OperatorEntry(
        name="depthwise_conv2d",
        build_fn=_build_depthwise_conv,
        shape_profiles=[{
            "label": "xcit_lpi",
            "input_shape": "1x768x14x14",
            "tensors": {},
        }],
        dtype="float32",
        cluster="B2",
    ),
}

DISPATCH_BASELINE_NAME = "dispatch_baseline"
DISPATCH_BASELINE_PATH = ONNX_GRAPHS_DIR / f"{DISPATCH_BASELINE_NAME}.onnx"


def list_operators() -> list[str]:
    return list(OPERATOR_REGISTRY.keys())


def get_entry(name: str) -> OperatorEntry:
    if name not in OPERATOR_REGISTRY:
        raise KeyError(f"Unknown operator: {name}. Known: {list_operators()}")
    return OPERATOR_REGISTRY[name]


def get_shape_profile(name: str, shape_index: int) -> dict[str, Any]:
    entry = get_entry(name)
    if shape_index < 0 or shape_index >= len(entry.shape_profiles):
        raise IndexError(
            f"shape_index {shape_index} out of range for {name} "
            f"(profiles={len(entry.shape_profiles)})"
        )
    return entry.shape_profiles[shape_index]


def list_profiles_for_block(block_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return isolated-op profiles for one SDPA block corner, keyed by operator name."""
    cfg = SDPA_BLOCK_CONFIG_BY_ID[block_id]
    return derive_isolated_profiles_for_block(cfg)


def build_operator_graph(name: str, shape_index: int = 0) -> tuple[Path, dict[str, Any], OperatorEntry]:
    entry = get_entry(name)
    profile = get_shape_profile(name, shape_index)
    path = _onnx_path(name, shape_index)
    meta = entry.build_fn(path, profile, OPSET)
    return path, meta, entry


def export_all_graphs(all_shapes: bool = False) -> None:
    ONNX_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[operators] ONNX helper opset: {OPSET}; "
        f"torch.onnx.export falls back to {FALLBACK_OPSET} on failure"
    )

    for name, entry in OPERATOR_REGISTRY.items():
        shape_indices = range(len(entry.shape_profiles)) if all_shapes else [0]
        for idx in shape_indices:
            path, _, _ = build_operator_graph(name, idx)
            print(f"  exported {path.name}")

    build_dispatch_baseline_graph(DISPATCH_BASELINE_PATH, OPSET)
    print(f"  exported {DISPATCH_BASELINE_PATH.name} (Add input+constant)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export operator ONNX graphs")
    parser.add_argument("--all-shapes", action="store_true", help="Export every shape profile")
    args = parser.parse_args()
    export_all_graphs(all_shapes=args.all_shapes)
