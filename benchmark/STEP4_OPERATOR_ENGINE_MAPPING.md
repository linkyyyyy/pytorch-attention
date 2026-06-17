# Step 4 — Operator Engine Mapping

> **PROVENANCE: energies from ANALYSIS_REFERENCE.md §3 (committed production snapshot), NOT primary results/ CSVs. Paper-grade table must be regenerated from tower CSVs (analysis_out.csv + analysis_out_20260615_145614_cpu_int8.csv) before publication.**

Operator-level deployment recommendations (avg corner). **No model-level claims.**

**Branch:** uProfAnalysis | **Energy source:** ANALYSIS_REFERENCE.md §3
**Config:** TIE_THRESHOLD=0.1, TIE_POLICY='report', PRECISION_GUARD=0.25 (advisory only), SIGN_MARGIN=0.1

## Methods / scope limitation

Complementary engine coverage asymmetry: **COMPUTE_BOUND_DENSE** ranks matched INT8 (cpu_INT8 vs npu_INT8; iGPU excluded). **MEMORY_BOUND** ranks FP32 (cpu_FP32 vs igpu_FP32; NPU excluded). CPU is the only engine in both grids. No operator receives a clean 3-way matched-precision ranking.

## Mapping table

| operator | shape_class | mechanism | winner | margin_pct | is_tie | architecture_ratio | sign_divergence | diverges_deployment | naive_winner | trust_flag |
|---|---|---|---|---|---|---|---|---|---|---|
| downsample_conv2d | avg | COMPUTE_BOUND_DENSE | npu | 1290.3226 | False | 14.1 | False | False | npu |  |
| avg_pool_token_mixer | avg | MEMORY_BOUND | igpu | 58.7209 | False |  | N/A | False | igpu | LOW-TRUST cpu_int8 (irrelevant to FP32 routing) |
| ffn_gemm[s2] | avg | COMPUTE_BOUND_DENSE | npu | 792.6829 | False | 8.92 | False | False | npu |  |
| ffn_gemm[s3] | avg | COMPUTE_BOUND_DENSE | npu | 601.9608 | False | 6.99 | False | False | npu |  |
| softmax[s1] | avg | MEMORY_BOUND | cpu | 321.3333 | False |  | True | False | cpu |  |
| qkv_proj_gemm[s5] | avg | COMPUTE_BOUND_DENSE | npu | 401.2531 | False | 5.02 | False | False | npu |  |
| qkv_proj_gemm[s3] | avg | COMPUTE_BOUND_DENSE | npu | 398.7277 | False | 5.0 | False | False | npu |  |
| qkv_proj_gemm[s4] | avg | COMPUTE_BOUND_DENSE | npu | 393.7028 | False | 4.95 | False | False | npu |  |
| out_proj_gemm[s1] | avg | COMPUTE_BOUND_DENSE | npu | 393.7028 | False | 4.93 | False | False | npu |  |
| depthwise_conv2d | avg | MEMORY_BOUND | tie (cpu/igpu) | 1.7621 | True |  | True | False | cpu |  |
| patch_embed_conv2d | avg | COMPUTE_BOUND_DENSE | npu | 198.5714 | False | 2.99 | False | False | npu |  |
| group_norm | avg | MEMORY_BOUND | tie (cpu/igpu) | 8.25 | True |  | False | True | npu |  |
| layer_norm[s1] | avg | MEMORY_BOUND | cpu | 104.7297 | False |  | False | False | cpu |  |
| gelu[s1] | avg | MEMORY_BOUND | cpu | 999.6169 | False |  | False | False | cpu |  |
| batch_norm | avg | MEMORY_BOUND | cpu | 95.098 | False |  | False | True | npu |  |
| attn_score_matmul[s1] | avg | COMPUTE_BOUND_DENSE | cpu | 29.1375 | False | 0.77 | False | False | cpu |  |
| residual_add[s1] | avg | MEMORY_BOUND | cpu | 236.0656 | False |  | False | False | cpu |  |
| xcit_cov_matmul | avg | COMPUTE_BOUND_DENSE | cpu | 98.2759 | False | 0.5 | False | False | cpu |  |
| attn_value_matmul[s1] | avg | COMPUTE_BOUND_DENSE | cpu | 130.0676 | False | 0.43 | False | False | cpu |  |
| sra_conv2d | avg | COMPUTE_BOUND_DENSE | EXCLUDED (npu INT8 untrusted) |  | False |  | N/A | N/A | igpu | UNTRUSTED npu INT8 |
| attn_block_fused[s1] | avg | COMPUTE_BOUND_DENSE | no matched recommendation (npu N/A) |  | False |  | N/A | N/A | igpu | N/A npu |

## Interpretation

### (a) GEMM / dense conv sweep → NPU (architectural)

Trusted **COMPUTE_BOUND_DENSE** rows with valid npu_INT8 recommend **npu** on matched-INT8 energy. `precision` < 1 on every dense GEMM/conv (INT8 penalizes CPU); `architecture_ratio` 3–14× on winners confirms the advantage is **architectural (XDNA2)**, not a quantization artifact. `sign_divergence=False` on those GEMMs (headline and best-INT8 agree: NPU).

### (b) sign_divergence pair: softmax + depthwise_conv2d

These reproduce the Step-3 decomposition trap — **headline winner ≠ best-INT8 engine** (`sign_divergence=True`). This is an INT8-vs-FP32 attribution question, **separate** from the deployment recommendation:

- **softmax[s1]:** gap=0.216 → headline **CPU** (cpu_FP32 < npu_INT8); arch=5.75 → best-INT8 **NPU**. Deployment (MEMORY_BOUND, FP32 grid) → **cpu** — same as naive.
- **depthwise_conv2d:** gap=0.515 → headline **CPU**; arch=3.15 → best-INT8 **NPU**. Deployment → **tie (cpu/igpu)** at 1.8%; `diverges_deployment=False` (naive cpu ∈ tie set).

**sign_divergence=True ops (SIGN_MARGIN=0.1):** softmax[s1], depthwise_conv2d — consolidation pair only.

**batch_norm footnote:** near-parity headline (gap=1.03) within sign margin; robust best-INT8 engine = CPU (arch=0.81). NOT contested-middle (it has a clear INT8 winner); excluded from the divergence pair on headline-noise grounds. `avg_pool_token_mixer` sign fields N/A (LOW-TRUST cpu_int8).

### (c) Contested middle → 10% rule ties

Attention matmuls where no engine dominates at matched INT8 (`attn_score_matmul`, `attn_value_matmul`, `xcit_cov_matmul`) route per mechanism. **10% deployment ties** at avg corner: **depthwise_conv2d** and **group_norm** (FP32 grid). TIE_POLICY=`report` for audit. (`batch_norm` is not in this band — see footnote above.)

For MEMORY_BOUND ops, arch/precision/gap values in the CSV are labelled decomposition cross-reference (§3) only — not ranking inputs.

### Trust exclusions

- **sra_conv2d:** EXCLUDED — npu INT8 UNTRUSTED
- **attn_block_fused:** no matched recommendation — npu N/A
- **avg_pool_token_mixer:** LOW-TRUST cpu_int8 irrelevant to FP32 routing
