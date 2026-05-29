from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt

# Ensure imports work when running this file directly.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.metrics import MetricsTracker
from simulation.engine import SimulationEngine
from simulation.environment import SimulationConfig


RUNS = 10
STEPS = 200


def build_config(kind: str) -> SimulationConfig:
    config = SimulationConfig()
    if kind == "resource_abundant":
        config.resource_regen_rate = 200
    elif kind == "uncapped_stability":
        config.stability_cap = 0.0
    return config


def run_single(
    config: SimulationConfig, steps: int, seed: int
) -> tuple[Dict[str, float], Dict[str, float], List[float]]:
    engine = SimulationEngine(config, seed=seed)
    metrics = MetricsTracker(config, max_history=steps)
    lineage_totals = {name: 0 for name in metrics.lineage_names}
    trait_totals = {
        "avg_replication_rate": 0.0,
        "avg_fidelity": 0.0,
        "avg_stability": 0.0,
    }
    diversity_series: List[float] = []

    for step in range(steps):
        engine.step()
        snapshot = metrics.update(step, engine.environment)
        for name in lineage_totals:
            lineage_totals[name] += metrics.last_lineage_counts.get(name, 0)
        trait_totals["avg_replication_rate"] += snapshot.avg_replication_rate
        trait_totals["avg_fidelity"] += snapshot.avg_fidelity
        trait_totals["avg_stability"] += snapshot.avg_stability
        diversity_series.append(snapshot.diversity)

    lineage_averages = {
        name: total / steps for name, total in lineage_totals.items()
    }
    trait_averages = {name: total / steps for name, total in trait_totals.items()}
    return lineage_averages, trait_averages, diversity_series


def run_config(
    config: SimulationConfig, runs: int, steps: int, seed_offset: int
) -> tuple[Dict[str, float], Dict[str, float], List[float]]:
    lineage_names = MetricsTracker(config, max_history=1).lineage_names
    lineage_totals = {name: 0.0 for name in lineage_names}
    trait_totals = {
        "avg_replication_rate": 0.0,
        "avg_fidelity": 0.0,
        "avg_stability": 0.0,
    }
    diversity_totals = [0.0 for _ in range(steps)]

    for run_index in range(runs):
        seed = seed_offset + run_index
        lineage_averages, trait_averages, diversity_series = run_single(
            config, steps, seed
        )
        for name in lineage_totals:
            lineage_totals[name] += lineage_averages.get(name, 0.0)
        for name in trait_totals:
            trait_totals[name] += trait_averages.get(name, 0.0)
        for index, value in enumerate(diversity_series):
            diversity_totals[index] += value

    lineage_results = {name: total / runs for name, total in lineage_totals.items()}
    trait_results = {name: total / runs for name, total in trait_totals.items()}
    diversity_results = [value / runs for value in diversity_totals]
    return lineage_results, trait_results, diversity_results


def plot_diversity(
    diversity_by_config: Dict[str, List[float]], output_path: Path
) -> None:
    steps = list(range(1, len(next(iter(diversity_by_config.values()))) + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, series in diversity_by_config.items():
        ax.plot(steps, series, label=label, linewidth=2)
    ax.set_title("Average Diversity Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Diversity")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, divider] + row_lines)


def main() -> None:
    configs = [
        ("Default", build_config("default")),
        ("Resource Abundant", build_config("resource_abundant")),
        ("Uncapped Stability", build_config("uncapped_stability")),
    ]

    lineage_names = MetricsTracker(configs[0][1], max_history=1).lineage_names
    results = {}
    trait_results = {}
    diversity_results = {}
    for index, (label, config) in enumerate(configs):
        lineage_res, trait_res, diversity_res = run_config(
            config, RUNS, STEPS, seed_offset=1000 + index * 100
        )
        results[label] = lineage_res
        trait_results[label] = trait_res
        diversity_results[label] = diversity_res

    headers = ["Lineage"] + [label for label, _ in configs]
    rows: List[List[str]] = []
    for name in lineage_names:
        rows.append(
            [
                name,
                f"{results['Default'][name]:.2f}",
                f"{results['Resource Abundant'][name]:.2f}",
                f"{results['Uncapped Stability'][name]:.2f}",
            ]
        )

    print(f"Runs: {RUNS}, Steps: {STEPS}")
    print(format_table(headers, rows))

    trait_headers = ["Trait"] + [label for label, _ in configs]
    trait_rows = [
        [
            "Avg Replication Rate",
            f"{trait_results['Default']['avg_replication_rate']:.3f}",
            f"{trait_results['Resource Abundant']['avg_replication_rate']:.3f}",
            f"{trait_results['Uncapped Stability']['avg_replication_rate']:.3f}",
        ],
        [
            "Avg Fidelity",
            f"{trait_results['Default']['avg_fidelity']:.3f}",
            f"{trait_results['Resource Abundant']['avg_fidelity']:.3f}",
            f"{trait_results['Uncapped Stability']['avg_fidelity']:.3f}",
        ],
        [
            "Avg Stability",
            f"{trait_results['Default']['avg_stability']:.3f}",
            f"{trait_results['Resource Abundant']['avg_stability']:.3f}",
            f"{trait_results['Uncapped Stability']['avg_stability']:.3f}",
        ],
    ]
    print("\nGlobal trait averages:")
    print(format_table(trait_headers, trait_rows))

    output_dir = SRC_ROOT.parent / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "diversity_avg.png"
    plot_diversity(diversity_results, output_path)
    print(f"\nDiversity chart saved to: {output_path}")


if __name__ == "__main__":
    main()
