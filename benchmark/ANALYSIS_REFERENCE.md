# Analysis Reference — Operator Energy Sweep

Slim working reference for the **uProfAnalysis** branch. Durable project context lives in
`PROJECT_CONTEXT.md` §7; this file is the analysis desk — tables, exclusions, metric rules, and
artifact paths for visualization work (heatmaps, regime plots, Step 4 mapping).

**Env:** `conda activate ryzen-ai-1.6.0` | **cwd:** `benchmark/`

---

## 1. Sessions

| Session | Engines | Idle (30 s) | Primary artifacts |
|---------|---------|-------------|-------------------|
| `20260612_143854` | cpu FP32, igpu FP32, npu XINT8 | 216.22 J | `results/runs_20260612_143854.csv`, `results/runs_enriched.csv`, `results/analysis_out.csv` |
| `20260615_145614_cpu_int8` | cpu_INT8 (QDQ on CPU EP) | 326.49 J | `results/runs_20260615_145614_cpu_int8.csv`, `results/analysis_out_20260615_145614_cpu_int8.csv` |

uProf traces: `results/uprof/<session_id>/` (gitignored). Power PNGs: `results/plots/`.

**Headline metric:** idle-subtracted `energy_per_op_J_mean` = `(window − idle_session) / iterations`.

---

## 2. Ratio definitions (cpu_INT8 control)

Methodological control only — not a deployment path.

| Ratio | Formula | Isolates |
|-------|---------|----------|
| `precision_ratio` | cpu_FP32 / cpu_INT8 | 8-bit vs 32-bit on same silicon |
| `architecture_ratio` | cpu_INT8 / npu_INT8 | Engine at matched `_xint8.onnx` graph |
| `original_gap` | cpu_FP32 / npu_INT8 | Cross-engine headline |

**Identity:** `precision × architecture = original_gap` → 1.000 by algebra (**consistency check only**;
`cpu_INT8` cancels). Validity rests on static-offset power traces + per-session idle subtraction (§5).

---

## 3. Decomposition table

Regenerated from `analysis_out.csv` + `analysis_out_20260615_145614_cpu_int8.csv`.
Energy in J/op (idle-subtracted mean). Sorted by `architecture_ratio` descending.

| Operator | cpu_FP32 | cpu_INT8 | npu_INT8 | igpu_FP32 | precision | arch | gap |
|----------|----------|----------|----------|-----------|-----------|------|-----|
| downsample_conv2d | 0.291 | 0.431 | 0.031 | 0.125 | 0.675 | 14.10 | 9.51 |
| avg_pool_token_mixer † | 0.00273 | 0.0499 | 0.00362 | 0.00172 | 0.055 | 13.79 | 0.755 |
| ffn_gemm[s2] | 0.0422 | 0.0732 | 0.00820 | 0.0299 | 0.576 | 8.92 | 5.14 |
| ffn_gemm[s3] | 0.0402 | 0.0716 | 0.0102 | 0.0286 | 0.562 | 6.99 | 3.93 |
| softmax[s1] | 0.00150 | 0.0401 | 0.00698 | 0.00632 | 0.038 | 5.75 | 0.216 |
| qkv_proj_gemm[s5] | 0.0115 | 0.0200 | 0.00399 | 0.0103 | 0.574 | 5.02 | 2.88 |
| qkv_proj_gemm[s3] | 0.0113 | 0.0196 | 0.00393 | 0.00958 | 0.575 | 5.00 | 2.88 |
| qkv_proj_gemm[s4] | 0.0115 | 0.0196 | 0.00397 | 0.0103 | 0.585 | 4.95 | 2.89 |
| out_proj_gemm[s1] | 0.0115 | 0.0196 | 0.00397 | 0.00958 | 0.585 | 4.93 | 2.89 |
| depthwise_conv2d | 0.00227 | 0.0139 | 0.00440 | 0.00231 | 0.164 | 3.15 | 0.515 |
| patch_embed_conv2d | 0.0110 | 0.0209 | 0.00700 | 0.0102 | 0.524 | 2.99 | 1.57 |
| group_norm | 0.00433 | 0.00443 | 0.00240 | 0.00400 | 0.977 | 1.85 | 1.81 |
| layer_norm[s1] | 0.00148 | 0.00135 | 0.00153 | 0.00303 | 1.10 | 0.88 | 0.968 |
| gelu[s1] | 0.00261 | 0.0133 | 0.0153 | 0.0287 | 0.196 | 0.87 | 0.171 |
| batch_norm | 0.00204 | 0.00161 | 0.00199 | 0.00398 | 1.27 | 0.81 | 1.03 |
| attn_score_matmul[s1] | 0.00315 | 0.00429 | 0.00554 | 0.00566 | 0.735 | 0.77 | 0.569 |
| residual_add[s1] | 0.000610 | 0.00220 | 0.00333 | 0.00205 | 0.278 | 0.66 | 0.183 |
| xcit_cov_matmul | 0.00136 | 0.00174 | 0.00345 | 0.00447 | 0.780 | 0.50 | 0.393 |
| attn_value_matmul[s1] | 0.00303 | 0.00296 | 0.00681 | 0.00717 | 1.02 | 0.43 | 0.445 |
| sra_conv2d | 0.187 | 0.524 | 0.455 ‡ | 0.127 | 0.357 | EXCLUDED | 0.411 |
| attn_block_fused[s1] | 0.0549 | 0.129 | — | 0.0413 | 0.427 | N/A | — |

† LOW-TRUST on cpu_int8 (CV 20.7%) — flag only, exclude from roll-up ranges.  
‡ npu UNTRUSTED — exclude from NPU ranking.

---

## 4. Regime buckets

### (a) Architectural INT8 wins — GEMMs + heavy conv

`ffn_gemm`, `qkv_proj_gemm`, `out_proj_gemm`, `downsample_conv2d`, `patch_embed_conv2d`, `depthwise_conv2d`

- `precision_ratio`: **0.16 – 0.67** (INT8 penalizes CPU)
- `architecture_ratio`: **2.99 – 14.10**
- **Read:** NPU advantage is architectural (XDNA2), not a quantization artifact.

### (b) Cheap memory-bound

`softmax`, `gelu`, `residual_add` (trusted roll-up)

- `precision_ratio`: **0.038 – 0.28**
- **Read:** QDQ-on-CPU is catastrophic; do not quantize these for a CPU path.
- `avg_pool_token_mixer` — same story qualitatively but LOW-TRUST; report separately.

### (c) Contested middle

`attn_score_matmul`, `attn_value_matmul`, `layer_norm`, `group_norm`, `batch_norm`, `xcit_cov_matmul`

- `precision_ratio`: **0.73 – 1.27**
- `architecture_ratio`: **0.43 – 1.85**
- **Read:** shape-dependent; neither precision nor engine dominates cleanly.

---

## 5. Sign-divergence (gap < 1 but arch > 1)

Headline FP32 winner ≠ best INT8 engine. Trusted cases:

| Operator | gap | arch | Note |
|----------|-----|------|------|
| softmax[s1] | 0.22 | 5.75 | cpu FP32 beats npu INT8; NPU still wins at matched INT8 |
| depthwise_conv2d | 0.52 | 3.15 | same pattern |

Also `avg_pool_token_mixer` (LOW-TRUST) and `sra_conv2d` (npu EXCLUDED) show the sign pattern.

---

## 6. Trust exclusions (always apply)

| Cell | Status |
|------|--------|
| `sra_conv2d × npu` | **UNTRUSTED** — ~45 W CPU-class power, throughput collapse |
| `attn_block_fused × npu` | **N/A** — VitisAI fused crash |
| `avg_pool_token_mixer × cpu_int8` | **LOW-TRUST** — CV 20.7% |

Main sweep Gate 1: **0/62** LOW-TRUST.

**NPU dispatch:** use idle-subtracted headline only; ~9 negative `dispatch_energy_J` on npu rows expected.

---

## 7. Step 4 metric basis (scoping — not yet executed)

| Regime | Rank on |
|--------|---------|
| (a) GEMM / heavy conv | `architecture_ratio` / matched-INT8 energy |
| (b) Cheap memory-bound | FP32 grid for default deployment; `architecture_ratio` if quantize-on-NPU branch |
| (c) Contested | FP32 grid when \|precision − 1\| ≤ 0.25; else `architecture_ratio` for NPU-quantized branch |

**Trap:** do not map on `original_gap` where `precision_ratio` departs far from 1.

---

## 8. Reproduce / extend analysis

```powershell
conda activate ryzen-ai-1.6.0
cd benchmark

# Re-join uProf (if needed)
python parse_energy.py --runs results/runs_20260612_143854.csv --uprof-dir results/uprof/20260612_143854

# Aggregate
python analysis.py

# cpu_int8 session
python parse_energy.py --runs results/runs_20260615_145614_cpu_int8.csv --uprof-dir results/uprof/20260615_145614_cpu_int8
python analysis.py --runs results/runs_20260615_145614_cpu_int8_enriched.csv --out results/analysis_out_20260615_145614_cpu_int8.csv
```

**Enriched repeats** (`runs_enriched.csv`) for per-config CV% and heatmap inputs (operator × engine × shape).

---

## 9. Visualization backlog (uProfAnalysis)

Suggested next features on this branch:

- **Heatmap:** operator × engine, `energy_per_op_J_mean` (log scale); facet by cluster or regime bucket
- **Ratio heatmap:** `architecture_ratio` and `precision_ratio` from §3
- **Sign-divergence scatter:** x = `original_gap`, y = `architecture_ratio`, label ops
- **Repeat variance:** CV% from `runs_enriched.csv` (flag > 15%)
- **Power trace gallery:** `results/plots/*_power.png` indexed by operator/engine
- **Fusion gap:** `analysis_out.csv` fusion_gap rows (cpu/igpu only; npu N/A)

Keep `results/` gitignored; commit analysis **scripts/notebooks** and this reference doc only.
