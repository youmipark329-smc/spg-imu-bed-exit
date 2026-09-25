# Ward-relevant IMU transition recognition — analysis code

Code to reproduce *"Sensor Modality Versus Pretraining for Recognising
Bed-Exit–Relevant Postural Transitions: A Gyroscope-Focused Feasibility Study on
Public Wearable-IMU Data"*, Park Y., Bold B., Cha W., *Sensors* 2026, 26(19),
6073, https://doi.org/10.3390/s26196073.

The study is a public-data feasibility benchmark. It defines a ward-relevant
ten-class task (four basic activities and six directional postural transitions)
on a waist-worn six-axis IMU, quantifies the contribution of the gyroscope by
ablating it and, as its mirror, the accelerometer, measures label efficiency,
compares hand-crafted features against a pretrained accelerometer-only motion
foundation model (UniMTS), and scores the deployable six-axis model as a bed-exit
alarm (probability calibration, alarm thresholds, event-level detection, and
windows that cross activity boundaries).

## Data (not redistributed here)

The dataset is public; download it and place it under `data/`:

- **HAPT** (Smartphone-Based Recognition of Human Activities and Postural
  Transitions), UCI Machine Learning Repository, dataset 341,
  https://doi.org/10.24432/C54G7M.

## Foundation model (not redistributed here)

The foundation-model steps (Section 3.5) use **UniMTS** (Zhang et al., NeurIPS
2024). Its source and released checkpoint are not redistributed here; obtain them
from the authors' repository and place them so that `external_UniMTS/` sits at
this repository's root:

```
external_UniMTS/
├── contrastive.py, model.py, ...      # from https://github.com/xiyuanzh/UniMTS
└── checkpoint/UniMTS.pth              # released checkpoint (~262 MB), https://huggingface.co/xiyuanz/UniMTS
```

`src/unimts_embed.py` and `src/run_finetune.py` import `contrastive` from this
folder; without it those two steps stop with `ModuleNotFoundError: No module
named 'contrastive'`, which is expected until UniMTS is placed. Everything else
runs without it. `src/check_unimts_ckpt.py` verifies that the checkpoint is the
released accelerometer-only (3-channel) model.

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

`IMU_reproduce_final.ipynb` reproduces the analyses of the original submission
end to end (Sections 3.1–3.5 and the label-efficiency analysis): the CPU steps
take a few minutes; the optional foundation-model steps need torch and a GPU.
The analyses added at revision (gyroscope-only ablation, classifier sensitivity,
alarm thresholds and calibration, event-level detection, boundary-crossing
windows) are run from the scripts listed below; they use the same windows,
folds, and seed as the notebook.

## Pipeline

Section, figure, and table numbers refer to the published article.

| Step | Script | Reported in | Output |
|------|--------|-------------|--------|
| Window HAPT + ward taxonomy | `src/prepare_hapt.py --wins 32 64 96 128` | Section 2.3, Table S1 | `results/hapt_windows_w{32,64,96,128}.npz` |
| Hand-crafted features (150 / 75 / 75) | `src/features.py` (imported; `use_gyro`, `use_acc`) | Section 2.4 | — |
| 12-class HAPT baseline + decomposition | `src/run_e1.py` | Section 3.1, Figure 2 | `e1_results.json`, per-class CSVs |
| 10-class ward task + confusion matrix | `src/run_transitions.py --wins 128` | Section 3.1, Figure 3 | confusion CSV |
| Sensor ablation: six-axis vs accelerometer-only vs gyroscope-only | `src/run_ablation.py --wins 128` | Section 3.2, Figure 4, Table 2 | `ablation_gyro.json`, `ablation_gyro_w128.csv` |
| Window-length sweep (survivorship-controlled) | `src/run_window_fair.py` | Section 3.3, Table 3, Figure S3 | `window_fair.csv` |
| Label efficiency, hand-crafted arms | `src/run_fm_compare.py --win 128` (no GPU needed) | Section 3.4, Figure 6 | `fm_label_efficiency_w128.csv` |
| Label efficiency, bed-exit bout counts | `src/run_label_efficiency.py --win 128` | Section 3.4 (bout counts) | `label_efficiency.csv` |
| UniMTS embeddings | `src/unimts_embed.py --win 128 --gyro 0 --padding 64` | — | `unimts_emb_w128.npz` |
| Representation comparison at full labels | `src/run_fm_compare.py --win 128` | Section 3.5, Table 4, Figure S1 | `fm_compare_w128.json`, `fm_per_class_w128.csv` |
| Full fine-tuning (three seeds) | `src/run_finetune.py --mode full --epochs 20 --padding 64 --seed {0,1,2} --tag _seed{0,1,2}`; `src/aggregate_finetune_seeds.py` | Section 3.5, Table 4 | fine-tune JSONs |
| Classifier sensitivity (random forest, logistic regression) | `src/run_classifier_sensitivity.py --win 128` | Section 3.5 | `classifier_sensitivity_w128.{json,csv}` |
| Alarm thresholds and probability calibration | `src/run_alarm_threshold.py --win 128` | Section 3.6 | `alarm_threshold_w128.{json,csv}` |
| Event-level detection and debounced false alarms | `src/run_event_level.py --win 128` | Section 3.7 | `event_level_w128.{json,csv}` |
| Boundary-crossing windows (continuous stream) | `src/run_boundary_windows.py --win 128 --label centre` (and `--label majority`) | Section 3.7 | `boundary_windows_w128_{centre,majority}.json` |
| Per-subject paired plot, window-sweep figure and table | `src/make_results_extras.py` | Figure 5, Figure S3, Table 3 | `fig_gyro_paired.png`, `fig_window_sweep.png`, `table_window_sweep.csv`, `gyro_paired_subjects.csv` |
| Figures 2, 4, 6 and the headline tables | `src/make_figures.py` | Figures 2, 4, 6; Tables 2, 4 | `figures/`, `table1_per_class.csv`, `table2_headline.csv` |
| Figure 3 (confusion matrix) | `src/make_extra_figures.py` | Figure 3 | `figures/` |
| Figure 1 (study pipeline) | `src/make_pipeline_figure.py` | Figure 1 | `fig_pipeline.{svg,pdf,eps,png,tif}` |
| Figure S1 (macro-F1 vs bed-exit F1 scatter) | `src/make_figS1.py` | Figure S1 | `figS1_representation_scatter.png` |
| Figure S2 (raw six-axis signals) | `src/make_raw_signal_figure.py` | Figure S2, Section 2.4 | `figS_raw_signals.png` |

Figure 1 is generated by `make_pipeline_figure.py` as a plain vector block
diagram (SVG, PDF, EPS, and 600 dpi PNG/TIFF); it is not a hand-designed graphic.

`run_fm_compare.py` produces two things: the full-label representation comparison
(Section 3.5) and the label-efficiency curves (Section 3.4). The macro-F1 and
bed-exit values in Section 3.4 are the `hand6+xgb` and `hand3+xgb` rows of
`fm_label_efficiency_w128.csv`, averaged over three seeds; each seed reseeds both
the label subsample and the classifier, and the bed-exit bout counts quoted in
Section 3.4 come from the same three seeds. `run_label_efficiency.py` is a
separate five-seed run of the same design, kept as a robustness check; its F1
values are not the ones reported and differ by up to 0.03.

`make_results_extras.py` recomputes the per-subject six-axis and
accelerometer-only scores rather than reading them from `ablation_gyro.json`,
which stores only the summary statistics; it reproduces the reported means
exactly (macro-F1 0.8225 vs 0.6467, bed-exit 0.6575 vs 0.3512) and writes the
per-subject values to `gyro_paired_subjects.csv` so that Figure 5 is auditable.

Figure S1 is drawn by `make_figS1.py` from `table2_headline.csv`, the headline
table written by `make_figures.py`, so it cannot drift from Table 4.

The alarm, event-level, and boundary-window analyses (Sections 3.6 and 3.7) use
`evaluate.cv_oof_proba`, which returns out-of-fold class probabilities from the
same folds and model as `cv_oof`; the argmax of those probabilities equals the
predictions used everywhere else. `run_boundary_windows.py` slides windows
continuously across the 642 contiguous stretches of the recordings and reports
interior and boundary-crossing windows separately; `--label centre` (the
reported setting) labels each window by its centre sample and `--label majority`
by its majority class (macro-F1 0.7537 vs 0.7538).

Only `unimts_embed.py` needs a GPU. Given `unimts_emb_w128.npz`, the frozen
probe and embedding readouts recompute on the CPU and reproduce the reported
values to four decimals. If `unimts_emb_w128.npz` is absent, `run_fm_compare.py`
runs the two hand-crafted arms only, which is enough for Section 3.4 and Figure
6, and writes its full-label output to `fm_compare_handonly_w128.json` so that
the four-arm artefacts behind Section 3.5 and Table 4 are never partially
overwritten.

Figures and tables are written into this repository's `figures/` and `results/`.
Evaluation utilities (subject-grouped CV, per-subject macro-F1, paired Wilcoxon,
rank-biserial effect size, bootstrap CI, Holm correction) are in
`src/evaluate.py`.

## Reproducibility notes

- All cross-validation is **subject-grouped** (no subject spans train/test).
- Statistics are **per-subject** (n = 30), paired across subjects; window
  overlap therefore cannot inflate the effective sample size.
- Random seeds are fixed; the released UniMTS checkpoint is accelerometer-only
  (3-channel), verified by `check_unimts_ckpt.py`.
- Every hand-crafted XGBoost result in the paper runs on the CPU histogram
  builder and reproduces to four decimals. `run_e1.py` accepts `--device cuda`
  for speed, but the CUDA builder is not bit-reproducible across runs and
  drivers, so it will not match the reported values exactly.
- Fine-tuning on GPU is not bit-reproducible; the paper reports the mean over
  three seeds (macro-F1 0.774 ± 0.019).

## Citation

Park, Y.; Bold, B.; Cha, W. Sensor Modality Versus Pretraining for Recognising
Bed-Exit–Relevant Postural Transitions: A Gyroscope-Focused Feasibility Study on
Public Wearable-IMU Data. *Sensors* **2026**, *26*, 6073.
https://doi.org/10.3390/s26196073. See `CITATION.cff`.
