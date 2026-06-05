# Operator & Architecture Selection — Energy DSE on AMD Ryzen AI 9 HX 370

**Goal:** design-space exploration of energy efficiency across the three on-die engines
(Zen 5 CPU · Radeon 890M iGPU · XDNA 2 NPU), measured at the **operator** level.
Architectures below are chosen as *sources* of a diverse, realistic operator set — the
unit of measurement is the operator, run on every engine.

All five primary models are available in `changzy00/pytorch-attention`
(`vision_transformers/`), which keeps the export harness uniform.

---

## 1. Selected architectures

### Primary set (locked — chosen for operator diversity, not accuracy)

| Model | Repo variant | Paper | Topology | One-line operator profile |
|---|---|---|---|---|
| **ViT** | `VisionTransformer` | arXiv 2010.11929 | Isotropic | Dense self-attention; full N×N attention matrix. The baseline. |
| **XCiT** | `xcit_nano_12_p16` | arXiv 2106.09681 | Isotropic | *Transposed* attention (channel-dim matrix) + depthwise-conv local mixing. |
| **PvT** | `pvt_t` | arXiv 2102.12122 | Hierarchical | Spatial-Reduction Attention: shrinks K/V via strided conv before attending. |
| **PoolFormer** | `poolformer_12` | arXiv 2111.11418 | Hierarchical | **No attention** — pooling is the token mixer in every block (the control). |
| **EfficientFormer** | `efficientformer_l1` | arXiv 2206.01191* | Hierarchical | Conv/pooling-heavy 4D stages; MHSA only in the final stage on a short sequence. |

\* The repo README links the V2 paper (2212.08059); `efficientformer_l1` is the original L1 model.

These five span the key axes: dense attention / transposed attention / reduced attention /
no attention / conv-hybrid-with-late-attention.

### Optional enrichment (only if we want to add a *genuinely new operator*)

| Model | Repo variant | Paper | Adds operator row |
|---|---|---|---|
| ShiftViT | `shift_t` | arXiv 2201.10801 | **Shift** — a zero-FLOP token mixer |
| MLP-Mixer | `MLP_Mixer` (`mlps/`) | arXiv 2105.01601 | **Token-mixing MLP** — attention-free but matmul-based (contrast to PoolFormer) |
| CSWin | `CSWin_64_12211_tiny_224` | arXiv 2107.00652 | **Windowed attention** — a local/windowed attention-matrix shape |

Everything else in the repo's ViT list mostly recombines operators already covered.

---

## 2. The three operator clusters

The distinction is **operator role within the block**, not which model it came from:

- **Cluster A — Global / structural:** the fixed transformer scaffolding, present across
  models regardless of how tokens are mixed. Does **not** form an attention matrix.
- **Cluster B1 — Token mixer (attention-matrix):** builds, normalizes, or consumes the
  attention matrix.
- **Cluster B2 — Token mixer (non-attention):** mixes across tokens but produces **no**
  attention matrix.

"Builds an attention matrix?" is the column that captures the attention-vs-not question.

---

## 3. Cluster A — Global / structural operators

| Operator | Role in the block | Models exercising it | Attention matrix? |
|---|---|---|---|
| Patch-embedding Conv2D | image → tokens (stem) | all 5 | No |
| Downsampling Conv2D | between-stage resolution reduction | PvT, EfficientFormer | No |
| Linear / GEMM (channel MLP / FFN) | per-token feed-forward | all 5 | No |
| GELU | activation | all 5 | No |
| Normalization | stabilization — **type varies**: LayerNorm (ViT/XCiT/PvT), GroupNorm (PoolFormer), BatchNorm (EfficientFormer 4D) | all 5 | No |
| Residual add | skip connections | all 5 | No |

*Note: Linear/GEMM also appears in the attention path (Q/K/V & output projections) — same
primitive, different shape; listed again in B1.*

---

## 4. Cluster B1 — Token mixer: attention-matrix operators

| Operator | Role | Models exercising it | Attention matrix? |
|---|---|---|---|
| Q/K/V + output projection GEMM | produce attention inputs / fuse outputs | ViT, XCiT, PvT, EfficientFormer (last stage) | Feeds it |
| Attention score matmul (QKᵀ) | **builds** the N×N matrix | ViT, PvT, EfficientFormer (last) | **Builds** |
| Cross-covariance matmul | **builds** a (head-dim × head-dim) channel matrix instead of N×N | XCiT | **Builds** |
| Softmax | normalizes the attention matrix | ViT, XCiT, PvT, EfficientFormer (last) | Operates on it |
| Attention·V matmul | **consumes** the matrix to reweight values | ViT, XCiT, PvT, EfficientFormer (last) | **Consumes** |
| Spatial-reduction Conv2D (SRA) | shrinks K/V before the matrix is built | PvT | Feeds it |

---

## 5. Cluster B2 — Token mixer: non-attention operators

| Operator | Role | Models exercising it | Attention matrix? |
|---|---|---|---|
| Average pooling (token mixer) | local token mixing, replaces attention | PoolFormer, EfficientFormer (4D) | No |
| Depthwise Conv2D | local token mixing (XCiT LPI; EF conv blocks) | XCiT, EfficientFormer | No |
| *(enrichment)* Shift | zero-FLOP token mixing | ShiftViT | No |
| *(enrichment)* Token-mixing MLP | transposed-MLP token mixing | MLP-Mixer | No |

---

## 6. How this maps to measurement

The clustering is for **interpretation**, not for measurement. Every operator above is
measured on **all three engines** (CPU / iGPU / NPU) as an isolated micro-benchmark:
single-op ONNX graph → ONNX Runtime with the chosen execution provider, run in a loop,
energy = average power × time (via uProf), idle baseline subtracted.

- Reading **down a column** (one engine, all operators) → that engine's per-operator energy profile.
- Reading **across a row** (one operator, three engines) → which engine is best specialized for that operator.
- The **cluster labels** then explain *model-level* differences: shared Cluster A ops cost
  the same everywhere, so any energy gap between two models is attributable to their
  differing B1/B2 mixer operators.
