# Attention-Block Dimension Extraction

> Config-extraction pass for the five shortlisted Vision Transformer architectures.
> All dimensions pulled from the actual model files and factory functions in this repo —
> **not** from paper defaults or external timm configs.
>
> **Input assumption:** 224×224 images (default `image_size` / `img_size` in every constructor).
>
> **Factory functions used (matching README + shortlist):**
> - `VisionTransformer()` — `vision_transformers/ViT.py`
> - `xcit_nano_12_p16()` — `vision_transformers/xcit.py`
> - `pvt_t()` — `vision_transformers/pvt.py`
> - `poolformer_12()` — `vision_transformers/poolformer.py`
> - `efficientformer_l1()` — `vision_transformers/efficientformer.py`

---

## Important code-vs-paper notes

### ViT

`VisionTransformer()` defaults to `embedding_dim=768`, `num_heads=4` → `head_dim=192`.
There is **no** `vit_b16()` factory in this repo. `PROJECT_CONTEXT.md` and
`benchmark/IMPLEMENTATION_BLUEPRINT.md` reference ViT-B/16 (12 heads, `head_dim=64), but
that is **not** what the constructor sets. Reported values below are from the code as written.

### XCiT

Two distinct attention types:

- **XCA** (12 blocks): cross-covariance attention. The score matrix is **`head_dim × head_dim`**
  (channel-axis), not `N × N`.
- **ClassAttention** (2 blocks): standard attention where only the CLS token queries all tokens.

### PvT

`N` in the table is the **query** token count entering the attention block. K and V are
spatially reduced by `sr_ratio` before the score matmul:

```
N_kv = (H / sr_ratio)²
```

### PoolFormer

**No attention blocks** — the token mixer is average pooling in every block. Per-stage `N` and `D`
are still reported because PoolFormer shares the same hierarchical pyramid topology as PvT
(useful for FFN / norm shape references and as the topology-matched control).

### EfficientFormer

Stages 1–3 use 4D `MetaBlock4D` (pooling token mixer, no MHSA). MHSA (`MetaBlock3D`) runs only
after stage 4, on the flattened 7×7 grid. Q/K projections use `query_dim=32`; V uses full
`dim=448` — asymmetric head dimensions.

---

## Attention-block dimension table

| architecture | stage | N | D | num_heads | head_dim |
|---|---|---:|---:|---:|---|
| **ViT** (`VisionTransformer()` default) | — | 197 | 768 | 4 | 192 |
| **XCiT-nano-12-p16** | XCA (12 blocks) | 196 | 128 | 4 | 32 |
| **XCiT-nano-12-p16** | ClassAttention (2 blocks) | 197 | 128 | 4 | 32 |
| **PvT-Tiny** (`pvt_t`) | 1 | 3136 | 64 | 1 | 64 |
| **PvT-Tiny** (`pvt_t`) | 2 | 784 | 128 | 2 | 64 |
| **PvT-Tiny** (`pvt_t`) | 3 | 196 | 320 | 5 | 64 |
| **PvT-Tiny** (`pvt_t`) | 4 | 49 | 512 | 8 | 64 |
| **PoolFormer-12** (`poolformer_12`) | 1 | 3136 | 64 | — | — |
| **PoolFormer-12** (`poolformer_12`) | 2 | 784 | 128 | — | — |
| **PoolFormer-12** (`poolformer_12`) | 3 | 196 | 320 | — | — |
| **PoolFormer-12** (`poolformer_12`) | 4 | 49 | 512 | — | — |
| **EfficientFormer-L1** (`efficientformer_l1`) | 1–3 (pooling) | — | 48 / 96 / 224 | — | — |
| **EfficientFormer-L1** (`efficientformer_l1`) | 4 (pooling, 4D) | 49 | 448 | — | — |
| **EfficientFormer-L1** (`efficientformer_l1`) | MHSA (3 blocks) | 49 | 448 | 8 | Q/K: **4**, V: **56** |

---

## Per-architecture derivation

### ViT — `VisionTransformer()` default

**Source:** `vision_transformers/ViT.py`

| Parameter | Value | Derivation |
|---|---|---|
| `image_size` | 224 | constructor default |
| `patch_size` | 16 | constructor default |
| `embedding_dim` (D) | 768 | constructor default |
| `num_heads` | 4 | constructor default |
| `depths` | 12 | constructor default |
| Patch grid | 14×14 | `grid_size = image_size // patch_size` |
| Patch tokens | 196 | `num_patches = grid_size²` |
| **N (with CLS)** | **197** | CLS concatenated in `forward()` before blocks |
| **head_dim** | **192** | `embedding_dim // num_heads = 768 // 4` |

```python
# ViT.py — PatchEmbedding defaults
image_size=224, patch_size=16, embedding_dim=768
grid_size = image_size // patch_size          # 14
num_patches = grid_size * grid_size           # 196

# ViT.py — VisionTransformer defaults
num_heads=4, embedding_dim=768, depths=12
# forward: x = cat([patches, cls_token]) → N = 197
```

---

### XCiT-nano-12-p16

**Source:** `vision_transformers/xcit.py` — `xcit_nano_12_p16()`

| Parameter | Value | Derivation |
|---|---|---|
| `img_size` | 224 | `XCiT` default |
| `patch_size` | 16 | factory arg |
| `embed_dim` (D) | 128 | factory arg |
| `depth` | 12 | factory arg (XCA blocks) |
| `num_heads` | 4 | factory arg |
| `cls_attn_layers` | 2 | `XCiT` default |
| Patch grid | 14×14 | `ConvPatchEmbed`: 4× stride-2 convs → ÷16 spatially |
| **N (XCA blocks)** | **196** | patch tokens only; no CLS during XCA |
| **N (ClassAttention)** | **197** | CLS prepended before `cls_attn_blocks` |
| **head_dim** | **32** | `128 // 4` |

XCA attention matrix shape per head: **32×32** (not 196×196).

```python
# xcit.py — factory function
def xcit_nano_12_p16(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=16, embed_dim=128, depth=12, num_heads=4, ...)
```

---

### PvT-Tiny — `pvt_t()`

**Source:** `vision_transformers/pvt.py`

| Parameter | Value |
|---|---|
| `dims` | [64, 128, 320, 512] |
| `num_heads` | [1, 2, 5, 8] |
| `sr_ratios` | [8, 4, 2, 1] |
| `depths` | [2, 2, 2, 2] |
| `mlp_ratios` | [8, 8, 4, 4] |

**Spatial progression** (224×224 input):

| Stage | Patch embed | H×W | N_q | D | heads | head_dim | sr_ratio | N_kv |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | 4×4, stride 4 | 56×56 | 3136 | 64 | 1 | 64 | 8 | 49 |
| 2 | 2×2, stride 2 | 28×28 | 784 | 128 | 2 | 64 | 4 | 49 |
| 3 | 2×2, stride 2 | 14×14 | 196 | 320 | 5 | 64 | 2 | 49 |
| 4 | 2×2, stride 2 | 7×7 | 49 | 512 | 8 | 64 | 1 | 49 |

QKᵀ score matrix shape per head at each stage: `[N_q × N_kv]`.

```python
# pvt.py — factory function
def pvt_t(num_classes=1000):
    return PVT(
        mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 2, 2],
        dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1], num_classes=num_classes)
```

---

### PoolFormer-12 — `poolformer_12()`

**Source:** `vision_transformers/poolformer.py`

No `Attention` module exists. Token mixer is `Pooling` (local average pool residual).

| Parameter | Value |
|---|---|
| `embedding_dims` | [64, 128, 320, 512] |
| `layers` | [2, 2, 6, 2] |
| `mlp_ratios` | [4, 4, 4, 4] |

**Spatial progression** (224×224 input):

| Stage | Downsample | H×W | N (H×W) | D |
|---|---|---|---:|---:|
| 1 | stem: 7×7, stride 4, pad 2 | 56×56 | 3136 | 64 |
| 2 | 3×3, stride 2, pad 1 | 28×28 | 784 | 128 |
| 3 | 3×3, stride 2, pad 1 | 14×14 | 196 | 320 |
| 4 | 3×3, stride 2, pad 1 | 7×7 | 49 | 512 |

Stem output size: `floor((224 + 2×2 − 7) / 4) + 1 = 56`.

```python
# poolformer.py — factory function
def poolformer_12(num_classes=1000):
    return PoolFormer(
        layers=[2, 2, 6, 2], embedding_dims=[64, 128, 320, 512],
        mlp_ratios=[4, 4, 4, 4])
```

---

### EfficientFormer-L1 — `efficientformer_l1()`

**Source:** `vision_transformers/efficientformer.py`

| Parameter | Value | Notes |
|---|---|---|
| `stem_dim` | 24 | factory override |
| `embed_dim` | [48, 96, 224, 448] | per-stage channel dim |
| `depths` | [3, 2, 6, 1] | MetaBlock4D counts per stage |
| `transformer_depth` | 3 | MetaBlock3D (MHSA) count |
| `num_heads` | 8 | constructor default (not overridden) |
| `query_dim` | 32 | constructor default (not overridden) |

**Spatial progression** (224×224 input):

| Stage | Downsample | H×W | N (H×W) | D | Token mixer |
|---|---|---|---:|---:|---|
| 1 | stem: 2× stride-2 convs | 56×56 | 3136 | 48 | pooling |
| 2 | 3×3, stride 2 | 28×28 | 784 | 96 | pooling |
| 3 | 3×3, stride 2 | 14×14 | 196 | 224 | pooling |
| 4 | 3×3, stride 2 | 7×7 | 49 | 448 | pooling |
| MHSA | flatten 7×7 → sequence | — | **49** | **448** | MHSA (3 blocks) |

MHSA head dimensions (asymmetric):

- Q/K `head_dim` = `query_dim // num_heads` = `32 // 8` = **4**
- V `head_dim` = `dim // num_heads` = `448 // 8` = **56**
- QKᵀ score matrix per head: **49×49**

```python
# efficientformer.py — factory + constructor defaults
def efficientformer_l1(num_classes=1000):
    return EfficientFormer(
        stem_dim=24, embed_dim=[48, 96, 224, 448],
        depths=[3, 2, 6, 1], transformer_depth=3)

# EfficientFormer.__init__ defaults (not overridden by l1):
num_heads=8, query_dim=32
```

---

## PvT spatial-reduction reference

For PvT-Tiny at 224×224, every stage with `sr_ratio > 1` reduces K/V to **49 tokens**
(7×7 spatial grid) regardless of query length:

| Stage | N_q | N_kv | QKᵀ shape per head |
|---|---:|---:|---|
| 1 | 3136 | 49 | 3136 × 49 |
| 2 | 784 | 49 | 784 × 49 |
| 3 | 196 | 49 | 196 × 49 |
| 4 | 49 | 49 | 49 × 49 |

---

## Proposed (N, D) block configs for SDPA sweeps

Three corners spanning the realistic range found in attention-bearing blocks above.
Each (N, D) pair is taken directly from a real block in the repo — not interpolated.

| label | N | D | num_heads | head_dim | drawn from |
|---|---:|---:|---:|---|---|
| **small** | 49 | 448 | 8 | Q/K=4, V=56 | **EfficientFormer-L1** MHSA (stage 4, 7×7 grid) — smallest sequence length with real MHSA in the shortlist |
| **avg** | 197 | 768 | 4 | 192 | **ViT** default (`VisionTransformer()`) — canonical isotropic full-attention block at mid-range N |
| **large** | 3136 | 64 | 1 | 64 | **PvT-Tiny** stage 1 — largest N in the repo; extreme tall-thin QKᵀ shape (with `N_kv=49`) |

### Rationale

- **N span:** 49 → 197 → 3136 covers the full range seen in attention blocks (7×7 late MHSA
  through 56×56 early PvT spatial-reduction attention).
- **D span:** 64 → 768 → 448 is not monotonic in D, but each corner is a real deployed pair:
  PvT stage 1 (low D, huge N), ViT default (high D, ~200 tokens), EfficientFormer MHSA
  (high D, tiny N).
- **Alternate avg candidate:** **PvT-Tiny stage 3** — (N=196, D=320, 5 heads, head_dim=64) —
  if a hierarchical mid-point is preferred with the same N as ViT patches but very different D
  and SR attention (`N_kv=49`).

### Not used as SDPA corners

| Architecture / block | Why excluded from small/avg/large |
|---|---|
| **XCiT XCA** (N=196, D=128) | Cross-covariance (channel-axis) attention — score matrix is 32×32, not N×N. Needs a separate operator profile, not standard SDPA-over-tokens. |
| **XCiT ClassAttention** (N=197, D=128) | CLS-only query; non-standard attention pattern. |
| **PoolFormer** (all stages) | No attention blocks. Useful as FFN/norm shape reference and PvT topology control, not SDPA corners. |
| **EfficientFormer stages 1–3** | Pooling only (4D feature maps); no MHSA. |

---

## Summary: N and D ranges across attention blocks

| Quantity | Min | Max | Where |
|---|---:|---:|---|
| N (query tokens) | 49 | 3136 | EfficientFormer MHSA / PvT stage 4 · PvT stage 1 |
| D (embed dim) | 64 | 768 | PvT stage 1 · ViT default |
| head_dim | 4 (Q/K) | 192 | EfficientFormer MHSA · ViT default |
| num_heads | 1 | 8 | PvT stage 1 · PvT stage 4 / EfficientFormer MHSA |

---

## Related files

| File | Relevance |
|---|---|
| `vision_transformers/ViT.py` | ViT defaults and `Attention` block |
| `vision_transformers/xcit.py` | `xcit_nano_12_p16`, `XCA`, `ClassAttention` |
| `vision_transformers/pvt.py` | `pvt_t`, spatial-reduction `Attention` |
| `vision_transformers/poolformer.py` | `poolformer_12`, pooling-only blocks |
| `vision_transformers/efficientformer.py` | `efficientformer_l1`, `MetaBlock3D` MHSA |
| `directives/operator_architecture_selection.md` | Shortlist rationale and operator clusters |
| `benchmark/IMPLEMENTATION_BLUEPRINT.md` | Operator registry shape profiles (note ViT-B/16 assumption vs code) |
| `PROJECT_CONTEXT.md` | §4 shortlist and Day 3/4 per-model operator notes |

---

*Generated: 2026-06-10. Config-extraction pass only — no harness or measurement code.*
