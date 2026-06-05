"""
operators.py — ONNX operator registry and graph export for energy benchmarking.

Each registry entry maps to (build_fn, shape_profiles, dtype, cluster).
Exports minimal single-operator graphs at a globally resolved ONNX opset (20, else 19).
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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


def resolve_global_opset() -> int:
    """Pick opset 20 globally; fall back to 19 if Gelu schema is unavailable at 20."""
    import onnx
    from onnx import defs

    for opset in (PREFERRED_OPSET, FALLBACK_OPSET):
        try:
            defs.get_schema("Gelu", opset, "")
            return opset
        except defs.SchemaError:
            continue
    return FALLBACK_OPSET


OPSET = resolve_global_opset()


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
    opset: int,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0],
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamo=False,
        )


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


def verify_gelu_graph(path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(path))
    gelu_nodes = [n for n in model.graph.node if n.op_type == "Gelu"]
    notes: list[str] = []
    if len(gelu_nodes) != 1:
        decomposed = [n.op_type for n in model.graph.node]
        msg = (
            f"gelu.onnx: expected exactly one Gelu node at opset {OPSET}, "
            f"found {len(gelu_nodes)}; node types={decomposed}. "
            "Gelu is native only at opset 20 — graph may have decomposed."
        )
        warnings.warn(msg, stacklevel=2)
        notes.append(msg)
    return notes


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


def _build_ffn_gemm(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    m, k, n = 197, 768, 3072
    model = nn.Linear(k, n, bias=False).eval()
    x = torch.randn(m, k)
    export_torch_module(model, path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _build_gelu(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    class GeluModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(x)

    x = torch.randn(197, 3072)
    export_torch_module(GeluModule().eval(), path, (x,), ["input"], ["output"], opset)
    verify_gelu_graph(path)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


def _build_layer_norm(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    model = nn.LayerNorm(768).eval()
    x = torch.randn(197, 768)
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


def _build_qkv_proj(path: Path, profile: dict[str, Any], opset: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    model = nn.Linear(768, 768, bias=False).eval()
    x = torch.randn(197, 768)
    export_torch_module(model, path, (x,), ["input"], ["output"], opset)
    return {"feeds": {"input": x.numpy().astype(np.float32)}, "output_names": ["output"]}


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

H = 12
N = 197
HEAD_DIM = 64

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
        shape_profiles=[{
            "label": "vit_ffn_expand",
            "input_shape": "197x768@768x3072",
            "tensors": {},
        }],
        dtype="float32",
        cluster="A",
    ),
    "gelu": OperatorEntry(
        name="gelu",
        build_fn=_build_gelu,
        shape_profiles=[{
            "label": "ffn_activation",
            "input_shape": "197x3072",
            "tensors": {},
        }],
        dtype="float32",
        cluster="A",
    ),
    "layer_norm": OperatorEntry(
        name="layer_norm",
        build_fn=_build_layer_norm,
        shape_profiles=[{
            "label": "vit_layernorm",
            "input_shape": "197x768",
            "tensors": {},
        }],
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
        shape_profiles=[{
            "label": "skip_connection",
            "input_shape": "197x768+197x768",
            "tensors": {"shape": [197, 768]},
        }],
        dtype="float32",
        cluster="A",
    ),
    "qkv_proj_gemm": OperatorEntry(
        name="qkv_proj_gemm",
        build_fn=_build_qkv_proj,
        shape_profiles=[{
            "label": "attn_projection",
            "input_shape": "197x768@768x768",
            "tensors": {},
        }],
        dtype="float32",
        cluster="B1",
    ),
    "attn_score_matmul": OperatorEntry(
        name="attn_score_matmul",
        build_fn=_build_attn_score_matmul,
        shape_profiles=[{
            "label": "vit_mha_qkt",
            "input_shape": f"{H}x{N}x{HEAD_DIM}@{H}x{HEAD_DIM}x{N}",
            "tensors": {"a_shape": [H, N, HEAD_DIM], "b_shape": [H, HEAD_DIM, N]},
        }],
        dtype="float32",
        cluster="B1",
    ),
    "xcit_cov_matmul": OperatorEntry(
        name="xcit_cov_matmul",
        build_fn=_build_xcit_cov_matmul,
        shape_profiles=[{
            "label": "xcit_cross_cov",
            "input_shape": f"{H}x{HEAD_DIM}x{N}@{H}x{N}x{HEAD_DIM}",
            "tensors": {"a_shape": [H, HEAD_DIM, N], "b_shape": [H, N, HEAD_DIM]},
        }],
        dtype="float32",
        cluster="B1",
    ),
    "softmax": OperatorEntry(
        name="softmax",
        build_fn=_build_softmax,
        shape_profiles=[
            {
                "label": "vit_attn_scores",
                "input_shape": f"{H}x{N}x{N}",
                "tensors": {"shape": [H, N, N], "axis": -1},
            },
            {
                "label": "xcit_channel_matrix",
                "input_shape": f"{H}x{HEAD_DIM}x{HEAD_DIM}",
                "tensors": {"shape": [H, HEAD_DIM, HEAD_DIM], "axis": -1},
            },
        ],
        dtype="float32",
        cluster="B1",
    ),
    "attn_value_matmul": OperatorEntry(
        name="attn_value_matmul",
        build_fn=_build_attn_value_matmul,
        shape_profiles=[{
            "label": "vit_mha_av",
            "input_shape": f"{H}x{N}x{N}@{H}x{N}x{HEAD_DIM}",
            "tensors": {"a_shape": [H, N, N], "b_shape": [H, N, HEAD_DIM]},
        }],
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


def build_operator_graph(name: str, shape_index: int = 0) -> tuple[Path, dict[str, Any], OperatorEntry]:
    entry = get_entry(name)
    profile = get_shape_profile(name, shape_index)
    path = _onnx_path(name, shape_index)
    meta = entry.build_fn(path, profile, OPSET)
    return path, meta, entry


def export_all_graphs(all_shapes: bool = False) -> None:
    ONNX_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[operators] Global ONNX opset: {OPSET} (preferred={PREFERRED_OPSET}, fallback={FALLBACK_OPSET})")
    if OPSET < PREFERRED_OPSET:
        warnings.warn(
            f"Using fallback opset {OPSET}; Gelu may decompose. Native Gelu requires opset 20.",
            stacklevel=2,
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
