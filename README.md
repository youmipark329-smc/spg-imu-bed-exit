# Ward-relevant IMU transition recognition — analysis code

Code to reproduce *"Sensor choice over model scale: gyroscope-dependent
recognition of ward bed-exit transitions from a wearable inertial measurement
unit (IMU), with implications for motion-induced arrhythmia false alarms"*
(submitted to *Sensors*).

The study is a public-data feasibility benchmark: it defines a ward-relevant
10-class task (four static postures + six directional postural transitions) on a
waist-worn 6-axis IMU, quantifies the gyroscope's contribution, measures label
efficiency, compares hand-crafted features against a pretrained motion foundation
model (UniMTS), and relates activity to chest-ECG signal quality.

## Data (not redistributed here)

Both datasets are public; download them and place under `data/`:

- **HAPT** (Smartphone-Based Recognition of Human Activities and Postural
  Transitions) — UCI Machine Learning Repository.
- **MHEALTH** (Mobile Health) — UCI Machine Learning Repository.

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

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| Window HAPT + ward taxonomy | `src/prepare_hapt.py --wins 128` | `results/hapt_windows_w128.npz` |
| Hand-crafted features | `src/features.py` (imported) | — |
| Gyroscope ablation (6- vs 3-axis) | `src/run_ablation.py --wins 128` | ablation JSON/CSV |
| 10-class transition task + confusion | `src/run_transitions.py --wins 128` | confusion CSV |
| Window-length sweep (survivorship-controlled) | `src/run_window_fair.py` | sweep CSV |
| Label efficiency | `src/run_label_efficiency.py --win 128` | label-efficiency CSV |
| UniMTS embeddings | `src/unimts_embed.py --win 128 --gyro 0 --padding 64` | embedding NPZ |
| Representation comparison | `src/run_fm_compare.py --win 128` | comparison JSON |
| Full fine-tuning | `src/run_finetune.py --mode full --epochs 20 --padding 64` | fine-tune JSON |
| ECG signal-quality bridge (MHEALTH) | `src/mhealth_ecg_bridge.py` | ECG SQI CSV + figure |
| Figures / tables | `src/make_figures.py`, `src/make_extra_figures.py` | `figures/`, table CSVs |

Evaluation utilities (subject-grouped CV, per-subject macro-F1, paired Wilcoxon,
bootstrap CI) are in `src/evaluate.py`. Released-UniMTS-checkpoint channel audit:
`src/check_unimts_ckpt.py`.

## Reproducibility notes

- All cross-validation is **subject-grouped** (no subject spans train/test).
- Statistics are **per-subject** (n = 30 for HAPT), paired across subjects.
- Random seeds are fixed; the released UniMTS checkpoint is accelerometer-only
  (3-channel), verified by `check_unimts_ckpt.py`.

## Citation

Please cite the *Sensors* article (details to be added upon publication).
