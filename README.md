# Ward-relevant IMU transition recognition — analysis code

Code to reproduce *"Sensor choice over model scale: gyroscope-dependent
recognition of ward bed-exit transitions from a wearable inertial measurement
unit (IMU)"*
(submitted to *Sensors*).

The study is a public-data feasibility benchmark: it defines a ward-relevant
10-class task (four static postures + six directional postural transitions) on a
waist-worn 6-axis IMU, quantifies the gyroscope's contribution, measures label
efficiency, compares hand-crafted features against a pretrained motion foundation
model (UniMTS).

## Data (not redistributed here)

The dataset is public; download it and place it under `data/`:

- **HAPT** (Smartphone-Based Recognition of Human Activities and Postural
  Transitions) — UCI Machine Learning Repository.

## Foundation model (not redistributed here)

The foundation-model steps (⑦–⑨ of the notebook; §3.5) use **UniMTS**
(Zhang et al., NeurIPS 2024). Its source and released checkpoint are not
redistributed here; obtain them from the authors' repository and place them so
that `external_UniMTS/` sits at this repository's root:

```
external_UniMTS/
├── contrastive.py, model.py, ...      # from https://github.com/xiyuanzh/UniMTS
└── checkpoint/UniMTS.pth              # released checkpoint (~262 MB)
```

`src/unimts_embed.py` and `src/run_finetune.py` import `contrastive` from this
folder; without it those two steps stop with `ModuleNotFoundError: No module
named 'contrastive'`, which is expected until UniMTS is placed. Steps ①–⑥
(§3.1–§3.4) do not use it. `src/check_unimts_ckpt.py` verifies the checkpoint is
the released accelerometer-only (3-channel) model.

## Environment

```bash
python -m venv venv && source venv/Scripts/activate   # Python 3.13
pip install -r requirements.txt
# Foundation-model steps only (optional):
pip install torch==2.13.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install huggingface_hub ftfy==6.3.1 regex git+https://github.com/openai/CLIP.git
```

Versions are pinned to those used for the reported numbers; results were
reproduced in a clean environment (hand-crafted/tree results bit-exact,
foundation-model embeddings bit-identical, fine-tuning macro-F1 within 0.002).

## Reproduction notebook

`IMU_reproduce_v2.ipynb` reproduces the **v2 paper (IMU only, no ECG)** end to end:
steps ①–⑥ run on CPU in a few minutes and produce §3.1–§3.4 with Figures 1–7 and
Table 2; the optional foundation-model steps ⑦–⑨ (torch + GPU) complete §3.5,
Tables 1 and 3, and Figure 8. Open it at the repository root and run the environment
and helper cells first. (`IMU_reproduce.ipynb` is the older combined notebook and is
superseded by the v2 version for the current paper.)

## Pipeline

| Step | Script | Reported in | Output |
|------|--------|-------------|--------|
| Window HAPT + ward taxonomy | `src/prepare_hapt.py --wins 32 64 96 128` | — | `results/hapt_windows_w{32,64,96,128}.npz` |
| Hand-crafted features | `src/features.py` (imported) | — | — |
| 12-class HAPT baseline + decomposition | `src/run_e1.py` | §3.1 (twelve-class), Figure 2 | `e1_results.json`, per-class CSVs |
| 10-class transition task + confusion | `src/run_transitions.py --wins 128` | §3.1 (ten-class), Figure 3 | confusion CSV |
| Gyroscope ablation (6- vs 3-axis) | `src/run_ablation.py --wins 128` | §3.2, Figure 4 | ablation JSON/CSV |
| Window-length sweep (survivorship-controlled) | `src/run_window_fair.py` | §3.3 | `window_fair.csv` |
| Label efficiency, hand-crafted arms | `src/run_fm_compare.py --win 128` (no GPU needed) | **§3.4, Figure 7** | `fm_label_efficiency_w128.csv` |
| Label efficiency, bed-exit bout counts | `src/run_label_efficiency.py --win 128` | §3.4 (bout counts only) | `label_efficiency.csv` |
| UniMTS embeddings | `src/unimts_embed.py --win 128 --gyro 0 --padding 64` | — | embedding NPZ |
| Representation comparison | `src/run_fm_compare.py --win 128` | §3.5, Table 3, Figure 8 | `fm_compare_w128.json` |
| Full fine-tuning | `src/run_finetune.py --mode full --epochs 20 --padding 64` | §3.5 | fine-tune JSON |
| Per-subject paired plot + window-sweep figure/table | `src/make_results_extras.py` | Figures 5–6, Table 2 | `fig_gyro_paired.png`, `fig_window_sweep.png`, `table_window_sweep.csv`, `gyro_paired_subjects.csv` |
| Figures / tables | `src/make_figures.py`, `src/make_extra_figures.py` | Figures 1–4, 7–8, S1, Tables 1, 3 | `figures/`, table CSVs |

`run_fm_compare.py` produces two things: the full-label representation comparison
(§3.5) and the label-efficiency curves (§3.4). The macro-F1 and bed-exit values in
§3.4 are the `hand6+xgb` and `hand3+xgb` rows of `fm_label_efficiency_w128.csv`,
averaged over three seeds; each seed reseeds both the label subsample and the
classifier, and the bed-exit bout counts quoted in §3.4 come from the same three
seeds. `run_label_efficiency.py` is a separate five-seed run of the same design,
kept as a robustness check; its F1 values are not the ones reported and differ by
up to 0.03.

Figures and tables are written into this repository's `figures/` and `results/`.
The manuscript build reads from `manuscript/analysis/figures` and
`analysis/tables` one level up, so run `../sync_manuscript_assets.py` after any
figure- or table-producing step and before rebuilding the DOCX. That script
reports which manuscript assets the repository cannot currently produce; with
`unimts_emb_w128.npz` in `results/` it should report none.

`make_results_extras.py` recomputes the per-subject six-axis and three-axis
scores rather than reading them from `ablation_gyro.json`, which stores only the
summary statistics; it reproduces the reported means exactly (macro-F1 0.8225 vs
0.6467, bed-exit 0.6575 vs 0.3512) and writes the per-subject values to
`gyro_paired_subjects.csv` so the figure is auditable.

Figure 8 (ranked dot plot) and Figure S1 (the same values as a macro-F1 versus
bed-exit F1 scatter) are both drawn by `make_figures.py` from the foundation-model
JSONs. If those JSONs are absent the two figures fall back to
`table2_headline.csv`, the archived table this same script wrote on an earlier
run, and say so on stdout; only the per-class artefacts (Table 1) strictly
require the JSONs.

Only `unimts_embed.py` needs a GPU. Given `unimts_emb_w128.npz`, the frozen
probe and embedding readouts recompute on the CPU and reproduce the reported
values to four decimals, including the paired statistics (frozen probe vs the
three-axis baseline: +0.084, 95% CI 0.048–0.122, p = 1.9 × 10⁻⁴).
Figure S1 is assembled into `manuscript/Supplementary_Materials.docx` by
`../build_supplementary_docx.py`, which shares the MDPI template with the main
manuscript builder.

The UniMTS embeddings are optional. If `unimts_emb_w128.npz` is absent,
`run_fm_compare.py` runs the two hand-crafted arms only — enough for §3.4 and
Figure 7 — and writes its full-label output to `fm_compare_handonly_w128.json`
so the four-arm artefacts behind §3.5, Table 3 and Figure 8 are never partially
overwritten. §3.1–§3.4 therefore need no GPU at all; only §3.5 does.

Evaluation utilities (subject-grouped CV, per-subject macro-F1, paired Wilcoxon,
bootstrap CI) are in `src/evaluate.py`. Released-UniMTS-checkpoint channel audit:
`src/check_unimts_ckpt.py`.

## Reproducibility notes

- All cross-validation is **subject-grouped** (no subject spans train/test).
- Statistics are **per-subject** (n = 30), paired across subjects.
- Random seeds are fixed; the released UniMTS checkpoint is accelerometer-only
  (3-channel), verified by `check_unimts_ckpt.py`.
- Every hand-crafted XGBoost result in the paper (§3.1–§3.3) runs on the CPU
  histogram builder and reproduces to four decimals. `run_e1.py` accepts
  `--device cuda` for speed, but the CUDA builder is not bit-reproducible
  across runs and drivers, so it will not match the reported values exactly.

## Citation

Please cite the *Sensors* article (details to be added upon publication).
