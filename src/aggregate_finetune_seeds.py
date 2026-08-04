"""Aggregate the multi-seed full-fine-tune runs (M2) into one summary.

Reads finetune_full_w128_seed{S}.json for each seed produced by
    run_finetune.py --mode full --seed S --tag _seedS
and reports, across seeds:
  - macro-F1 and bed-exit F1 as mean +/- SD (the between-seed variance the
    single-seed number could not show)
  - the paired-vs-6-axis macro-F1 difference and its p-value per seed, so the
    "6-axis beats the fine-tuned FM" claim can be stated as holding across seeds

It does NOT touch the canonical finetune_full_w128.json (the reported seed-0
artifact) or any figure; it only prints a summary and writes
finetune_full_w128_multiseed.json for the manuscript update to read.
"""
import json
import statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    seeds = []
    for p in sorted(RES.glob("finetune_full_w128_seed*.json")):
        seeds.append((p.name, json.loads(p.read_text())))
    if not seeds:
        print("no finetune_full_w128_seed*.json found — run M2 first.")
        return

    macro = [j["macro_f1_mean"] for _, j in seeds]
    bed = [j["bed_exit_f1_mean"] for _, j in seeds]

    def ms(xs):
        return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)

    m_mean, m_sd = ms(macro)
    b_mean, b_sd = ms(bed)

    print(f"seeds found: {len(seeds)}")
    for name, j in seeds:
        vs = j.get("vs_hand6_macro", {})
        print(f"  {name}: macro {j['macro_f1_mean']:.4f}  bed {j['bed_exit_f1_mean']:.4f}"
              f"   vs-hand6 diff {vs.get('mean_diff', float('nan')):+.4f}"
              f" p={vs.get('wilcoxon_p', float('nan')):.2e}")
    print()
    print(f"macro-F1   : {m_mean:.4f} +/- {m_sd:.4f}  (range {min(macro):.4f}-{max(macro):.4f})")
    print(f"bed-exit F1: {b_mean:.4f} +/- {b_sd:.4f}  (range {min(bed):.4f}-{max(bed):.4f})")

    # does 6-axis beat the fine-tuned FM in every seed?
    diffs = [j.get("vs_hand6_macro", {}).get("mean_diff") for _, j in seeds]
    ps = [j.get("vs_hand6_macro", {}).get("wilcoxon_p") for _, j in seeds]
    if all(d is not None for d in diffs):
        all_neg = all(d < 0 for d in diffs)          # FM below 6-axis
        all_sig = all(p is not None and p < 0.05 for p in ps)
        print()
        print(f"6-axis hand-crafted above fine-tuned FM in all {len(seeds)} seeds: "
              f"{'YES' if all_neg else 'NO'}"
              f" ({'all p<0.05' if all_sig else 'not all significant'})")

    # per-class F1 averaged across seeds (for Table 1 and the figures)
    classes = list(seeds[0][1]["per_class"].keys())
    per_class_mean = {c: st.mean([j["per_class"][c] for _, j in seeds])
                      for c in classes}

    out = {
        "n_seeds": len(seeds),
        "seeds": [j.get("seed") for _, j in seeds],
        "macro_f1_seed_mean": m_mean, "macro_f1_seed_sd": m_sd,
        "bed_exit_f1_seed_mean": b_mean, "bed_exit_f1_seed_sd": b_sd,
        "macro_f1_seed_min": min(macro), "macro_f1_seed_max": max(macro),
        "bed_exit_f1_seed_min": min(bed), "bed_exit_f1_seed_max": max(bed),
        "per_class_seed_mean": per_class_mean,
        "per_seed": [{"file": n, "macro": j["macro_f1_mean"],
                      "bed": j["bed_exit_f1_mean"],
                      "vs_hand6_macro": j.get("vs_hand6_macro")}
                     for n, j in seeds],
    }
    (RES / "finetune_full_w128_multiseed.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote -> {RES / 'finetune_full_w128_multiseed.json'}")


if __name__ == "__main__":
    main()
