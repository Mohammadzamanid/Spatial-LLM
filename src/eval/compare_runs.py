"""
src/eval/compare_runs.py

Compare two trained runs (e.g. neuroscience modules ON vs OFF) by reading the
final eval_loss from each run's trainer_state.json. This is the core experiment:
does the neuroscience stack actually improve held-out performance?

Usage:
    python -m src.eval.compare_runs --runs outputs/neuro_on outputs/neuro_off
"""
import argparse
import json
from pathlib import Path


def _final_eval_loss(run_dir: Path):
    """Read the lowest eval_loss recorded in a run's trainer_state.json."""
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        # HF may store it in a checkpoint subdir
        candidates = sorted(run_dir.glob("checkpoint-*/trainer_state.json"))
        if not candidates:
            raise FileNotFoundError(f"No trainer_state.json under {run_dir}")
        state_path = candidates[-1]
    state = json.loads(state_path.read_text())
    eval_losses = [
        e["eval_loss"] for e in state.get("log_history", []) if "eval_loss" in e
    ]
    if not eval_losses:
        raise ValueError(f"No eval_loss entries in {state_path}")
    return min(eval_losses), eval_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run output dirs to compare (label inferred from dir name)")
    args = ap.parse_args()

    results = {}
    for run in args.runs:
        run_dir = Path(run)
        label = run_dir.name
        try:
            best, history = _final_eval_loss(run_dir)
            results[label] = best
            print(f"\n[{label}]")
            print(f"  eval_loss history: {[round(x, 4) for x in history]}")
            print(f"  best eval_loss:    {best:.4f}")
        except (FileNotFoundError, ValueError) as e:
            print(f"\n[{label}]  could not read: {e}")

    if len(results) >= 2:
        print("\n" + "=" * 50)
        print("VERDICT")
        print("=" * 50)
        ranked = sorted(results.items(), key=lambda kv: kv[1])
        best_label, best_val = ranked[0]
        for label, val in ranked:
            marker = "  <-- best" if label == best_label else ""
            print(f"  {label:20s} eval_loss = {val:.4f}{marker}")
        if len(ranked) == 2:
            (a_label, a), (b_label, b) = ranked
            rel = (b - a) / b * 100
            print(f"\n  {a_label} is {rel:.1f}% lower eval_loss than {b_label}.")
            print("  (Lower = better. Whether this gap is meaningful depends on")
            print("   run-to-run variance — ideally repeat with 2-3 seeds.)")


if __name__ == "__main__":
    main()
