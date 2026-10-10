"""Re-render the paper result figures from the saved evaluation results.

``paper_4wis_eval.py`` regenerates these figures at the end of a full run, which
re-simulates every episode.  This helper replots only from the stored
``paper_pvo_results.npz`` so that purely cosmetic changes (for example the
AVO/AVO-UQ method naming) can be refreshed without repeating the experiments.
The plotting code mirrors ``save_outputs`` in ``paper_4wis_eval.py``.

Usage (from the repository root):
    python experiments/replot_paper_results.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = os.path.join("experiments", "paper_results")

# Mirror of METHOD_LABELS / METHOD_ORDER in paper_4wis_eval.py.
METHOD_LABELS = {
    "td3_diff": "DRL-VO",
    "4wis_mm": "PVO",
    "4wis_pvo": "AVO",
    "4wis_pvo_uq": "AVO-UQ",
}
METHOD_ORDER = ["td3_diff", "4wis_mm", "4wis_pvo", "4wis_pvo_uq"]


def main():
    npz_path = os.path.join(RESULT_DIR, "paper_pvo_results.npz")
    aggregate = np.load(npz_path, allow_pickle=True)["aggregate"].item()

    methods = METHOD_ORDER
    labels = [METHOD_LABELS[m] for m in methods]

    success = [aggregate[m]["success"]["mean"] * 100 for m in methods]
    collision = [aggregate[m]["collision"]["mean"] * 100 for m in methods]
    timeout = [aggregate[m]["timeout"]["mean"] * 100 for m in methods]
    success_sem = [aggregate[m]["success"]["sem"] * 100 for m in methods]
    collision_sem = [aggregate[m]["collision"]["sem"] * 100 for m in methods]
    timeout_sem = [aggregate[m]["timeout"]["sem"] * 100 for m in methods]
    x = np.arange(len(methods))
    width = 0.26

    plt.figure(figsize=(8.6, 4.4))
    plt.bar(x - width, success, width, yerr=success_sem, capsize=4, label="Success")
    plt.bar(x, collision, width, yerr=collision_sem, capsize=4, label="Collision")
    plt.bar(x + width, timeout, width, yerr=timeout_sem, capsize=4, label="Timeout")
    plt.xticks(x, labels, rotation=10)
    plt.ylabel("Rate (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "paper_pvo_main_rates.png"), dpi=200)
    plt.savefig(os.path.join(RESULT_DIR, "paper_4wis_rates.png"), dpi=200)
    plt.close()

    steps = [aggregate[m]["steps"]["mean"] for m in methods]
    distance = [aggregate[m]["distance"]["mean"] for m in methods]
    plt.figure(figsize=(8.6, 4.2))
    plt.plot(labels, steps, marker="o", label="Average steps")
    plt.plot(labels, distance, marker="s", label="Path length")
    plt.xticks(rotation=10)
    plt.ylabel("Episode metric")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "paper_pvo_efficiency.png"), dpi=200)
    plt.close()

    print(
        "Rewrote paper_pvo_main_rates.png, paper_4wis_rates.png "
        "and paper_pvo_efficiency.png with labels: " + ", ".join(labels)
    )


if __name__ == "__main__":
    main()
