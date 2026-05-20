from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

from simulation.environment import Environment, SimulationConfig
from simulation.replicator import Replicator

TRAIT_CLASSES = (
    "High Replication",
    "High Fidelity",
    "High Stability",
    "Balanced",
)


@dataclass
class MetricsSnapshot:
    step: int
    population: int
    resources: int
    diversity: int
    avg_replication_rate: float
    avg_fidelity: float
    avg_stability: float
    dominant_lineage: str
    dominant_trait_class: str


class MetricsTracker:
    def __init__(self, config: SimulationConfig, max_history: int = 300) -> None:
        self.max_history = max_history
        self.steps: Deque[int] = deque(maxlen=max_history)
        self.history: Dict[str, Deque[float]] = {
            "population": deque(maxlen=max_history),
            "resources": deque(maxlen=max_history),
            "diversity": deque(maxlen=max_history),
            "avg_replication_rate": deque(maxlen=max_history),
            "avg_fidelity": deque(maxlen=max_history),
            "avg_stability": deque(maxlen=max_history),
        }
        self.config = config
        self.lineage_names: List[str] = []
        self.trait_classes: List[str] = []
        self.lineage_history: Dict[str, Deque[int]] = {}
        self.trait_class_history: Dict[str, Deque[int]] = {}
        self.last_snapshot: MetricsSnapshot | None = None
        self.last_lineage_counts: Dict[str, int] = {}
        self.last_trait_class_counts: Dict[str, int] = {}
        self._trait_ranges: Tuple[
            Tuple[float, float], Tuple[float, float], Tuple[float, float]
        ]
        self.reconfigure(config)

    def reconfigure(self, config: SimulationConfig) -> None:
        self.config = config
        self.lineage_names = [preset.name for preset in config.lineage_presets]
        if "Unclassified" not in self.lineage_names:
            self.lineage_names.append("Unclassified")
        self.trait_classes = list(TRAIT_CLASSES)
        self.lineage_history = {
            name: deque(maxlen=self.max_history) for name in self.lineage_names
        }
        self.trait_class_history = {
            name: deque(maxlen=self.max_history) for name in self.trait_classes
        }
        self._trait_ranges = self._compute_trait_ranges(config.lineage_presets)
        self.reset()

    def reset(self) -> None:
        self.steps.clear()
        for series in self.history.values():
            series.clear()
        for series in self.lineage_history.values():
            series.clear()
        for series in self.trait_class_history.values():
            series.clear()
        self.last_snapshot = None
        self.last_lineage_counts = {}
        self.last_trait_class_counts = {}

    def update(self, step: int, env: Environment) -> MetricsSnapshot:
        population = len(env.replicators)
        resources = env.resources
        diversity = self._compute_diversity(env.replicators)
        avg_rep = self._avg([rep.replication_rate for rep in env.replicators])
        avg_fid = self._avg([rep.fidelity for rep in env.replicators])
        avg_stab = self._avg([rep.stability for rep in env.replicators])
        lineage_counts = {name: 0 for name in self.lineage_names}
        trait_class_counts = {name: 0 for name in self.trait_classes}

        for rep in env.replicators:
            lineage_name = (
                rep.lineage_name
                if rep.lineage_name in lineage_counts
                else "Unclassified"
            )
            lineage_counts[lineage_name] += 1
            trait_class = self._classify_traits(rep)
            trait_class_counts[trait_class] += 1

        self.steps.append(step)
        self.history["population"].append(population)
        self.history["resources"].append(resources)
        self.history["diversity"].append(diversity)
        self.history["avg_replication_rate"].append(avg_rep)
        self.history["avg_fidelity"].append(avg_fid)
        self.history["avg_stability"].append(avg_stab)

        for name, count in lineage_counts.items():
            self.lineage_history[name].append(count)
        for name, count in trait_class_counts.items():
            self.trait_class_history[name].append(count)

        dominant_lineage = self._dominant_key(lineage_counts)
        dominant_trait = self._dominant_key(trait_class_counts)

        self.last_lineage_counts = lineage_counts
        self.last_trait_class_counts = trait_class_counts

        snapshot = MetricsSnapshot(
            step=step,
            population=population,
            resources=resources,
            diversity=diversity,
            avg_replication_rate=avg_rep,
            avg_fidelity=avg_fid,
            avg_stability=avg_stab,
            dominant_lineage=dominant_lineage,
            dominant_trait_class=dominant_trait,
        )
        self.last_snapshot = snapshot
        return snapshot

    def series(self, name: str) -> List[float]:
        if name in self.history:
            return list(self.history.get(name, []))
        if name in self.lineage_history:
            return list(self.lineage_history.get(name, []))
        if name in self.trait_class_history:
            return list(self.trait_class_history.get(name, []))
        return []

    def step_series(self) -> List[int]:
        return list(self.steps)

    @staticmethod
    def _compute_diversity(replicators: List[Replicator]) -> int:
        signatures = {rep.signature() for rep in replicators}
        return len(signatures)

    def _classify_traits(self, rep: Replicator) -> str:
        rep_range, fid_range, stab_range = self._trait_ranges

        rep_norm = self._normalize(rep.replication_rate, rep_range)
        fid_norm = self._normalize(rep.fidelity, fid_range)
        stab_norm = self._normalize(rep.stability, stab_range)

        norms = {
            "High Replication": rep_norm,
            "High Fidelity": fid_norm,
            "High Stability": stab_norm,
        }
        spread = max(norms.values()) - min(norms.values())
        if spread <= self.config.trait_class_balance_threshold:
            return "Balanced"
        return max(norms.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _normalize(value: float, bounds: Tuple[float, float]) -> float:
        min_val, max_val = bounds
        if max_val - min_val <= 1e-6:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    @staticmethod
    def _compute_trait_ranges(
        presets,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        if not presets:
            return (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
        rep_values = [preset.replication_rate for preset in presets]
        fid_values = [preset.fidelity for preset in presets]
        stab_values = [preset.stability for preset in presets]
        return (
            (min(rep_values), max(rep_values)),
            (min(fid_values), max(fid_values)),
            (min(stab_values), max(stab_values)),
        )

    @staticmethod
    def _dominant_key(counts: Dict[str, int]) -> str:
        if not counts:
            return "None"
        return max(counts.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _avg(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)
