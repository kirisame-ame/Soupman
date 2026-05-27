from dataclasses import dataclass, field
from typing import List, Tuple
import random

from .replicator import Replicator


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def enforce_trait_total(
    replication_rate: float,
    fidelity: float,
    stability: float,
    total_cap: float,
) -> Tuple[float, float, float]:
    if total_cap <= 0.0:
        return replication_rate, fidelity, stability
    total = replication_rate + fidelity + stability
    if total <= 0.0 or total <= total_cap:
        return replication_rate, fidelity, stability
    scale = total_cap / total
    return (
        clamp(replication_rate * scale),
        clamp(fidelity * scale),
        clamp(stability * scale),
    )


def apply_stability_cap(stability: float, cap: float) -> float:
    if cap <= 0.0:
        return clamp(stability)
    return clamp(min(stability, cap))


def apply_stability_replication_tradeoff(
    replication_rate: float,
    stability: float,
    stability_floor: float,
    strength: float,
) -> float:
    if strength <= 0.0:
        return replication_rate
    excess = max(0.0, stability - stability_floor)
    penalty = min(1.0, excess * strength)
    return clamp(replication_rate * (1.0 - penalty))


@dataclass(frozen=True)
class LineagePreset:
    name: str
    replication_rate: float
    fidelity: float
    stability: float
    weight: float = 1.0


def default_lineage_presets() -> List[LineagePreset]:
    return [
        LineagePreset(
            name="High Replication",
            replication_rate=0.70,
            fidelity=0.70,
            stability=0.64,
            weight=0.2,
        ),
        LineagePreset(
            name="High Fidelity",
            replication_rate=0.34,
            fidelity=0.96,
            stability=0.88,
            weight=0.2,
        ),
        LineagePreset(
            name="High Stability",
            replication_rate=0.24,
            fidelity=0.86,
            stability=0.93,
            weight=0.2,
        ),
        LineagePreset(
            name="Balanced",
            replication_rate=0.32,
            fidelity=0.88,
            stability=0.90,
            weight=0.4,
        ),
    ]


@dataclass
class SimulationConfig:
    initial_population: int = 80
    initial_resources: int = 2000
    max_resources: int = 2000
    min_sequence_length: int = 6
    max_sequence_length: int = 12
    alphabet_size: int = 4
    base_replication_rate: float = 0.32
    base_fidelity: float = 0.85
    base_stability: float = 0.92
    replication_rate_jitter: float = 0.02
    fidelity_jitter: float = 0.01
    stability_jitter: float = 0.01
    mutation_multiplier: float = 1.0
    trait_mutation_rate: float = 0.25
    trait_mutation_strength: float = 0.05
    trait_total_cap: float = 2.2
    stability_replication_tradeoff_strength: float = 1.6
    stability_replication_tradeoff_floor: float = 0.70
    stability_cap: float = 0.93
    replication_multiplier: float = 1.0
    stability_multiplier: float = 1.0
    resource_regen_rate: int = 0
    step_interval: float = 0.05
    lineage_presets: List[LineagePreset] = field(
        default_factory=default_lineage_presets
    )
    lineage_trait_jitter: float = 0.01
    trait_class_balance_threshold: float = 0.15


class Environment:
    def __init__(self, config: SimulationConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.resources = config.initial_resources
        self.max_resources = config.max_resources
        self.replicators: List[Replicator] = []
        self._next_uid = 1
        self._seed_initial_population()

    def _seed_initial_population(self) -> None:
        preset_entries = [
            (index, preset)
            for index, preset in enumerate(self.config.lineage_presets)
            if preset.weight > 0
        ]
        total_weight = sum(preset.weight for _, preset in preset_entries)

        for _ in range(self.config.initial_population):
            sequence_length = self.rng.randint(
                self.config.min_sequence_length, self.config.max_sequence_length
            )
            sequence = [
                self.rng.randrange(self.config.alphabet_size)
                for _ in range(sequence_length)
            ]
            if preset_entries and total_weight > 0:
                lineage_id, preset = self._choose_preset(preset_entries, total_weight)
                replication_rate, fidelity, stability = self._apply_preset(preset)
                lineage_name = preset.name
            else:
                lineage_id = -1
                lineage_name = "Unclassified"
                replication_rate = clamp(
                    self.config.base_replication_rate
                    + self.rng.uniform(
                        -self.config.replication_rate_jitter,
                        self.config.replication_rate_jitter,
                    )
                )
                fidelity = clamp(
                    self.config.base_fidelity
                    + self.rng.uniform(
                        -self.config.fidelity_jitter, self.config.fidelity_jitter
                    )
                )
                stability = clamp(
                    self.config.base_stability
                    + self.rng.uniform(
                        -self.config.stability_jitter, self.config.stability_jitter
                    )
                )
                replication_rate, fidelity, stability = enforce_trait_total(
                    replication_rate,
                    fidelity,
                    stability,
                    self.config.trait_total_cap,
                )
                stability = apply_stability_cap(
                    stability,
                    self.config.stability_cap,
                )
                replication_rate = apply_stability_replication_tradeoff(
                    replication_rate,
                    stability,
                    self.config.stability_replication_tradeoff_floor,
                    self.config.stability_replication_tradeoff_strength,
                )
            self.replicators.append(
                Replicator(
                    uid=self._allocate_uid(),
                    sequence=sequence,
                    replication_rate=replication_rate,
                    fidelity=fidelity,
                    stability=stability,
                    lineage_id=lineage_id,
                    lineage_name=lineage_name,
                )
            )

    def _choose_preset(
        self, preset_entries: List[Tuple[int, LineagePreset]], total_weight: float
    ) -> Tuple[int, LineagePreset]:
        roll = self.rng.uniform(0.0, total_weight)
        upto = 0.0
        for lineage_id, preset in preset_entries:
            upto += preset.weight
            if roll <= upto:
                return lineage_id, preset
        return preset_entries[-1]

    def _apply_preset(self, preset: LineagePreset) -> Tuple[float, float, float]:
        jitter = self.config.lineage_trait_jitter
        replication_rate = clamp(
            preset.replication_rate + self.rng.uniform(-jitter, jitter)
        )
        fidelity = clamp(preset.fidelity + self.rng.uniform(-jitter, jitter))
        stability = clamp(preset.stability + self.rng.uniform(-jitter, jitter))
        replication_rate, fidelity, stability = enforce_trait_total(
            replication_rate,
            fidelity,
            stability,
            self.config.trait_total_cap,
        )
        stability = apply_stability_cap(
            stability,
            self.config.stability_cap,
        )
        replication_rate = apply_stability_replication_tradeoff(
            replication_rate,
            stability,
            self.config.stability_replication_tradeoff_floor,
            self.config.stability_replication_tradeoff_strength,
        )
        return replication_rate, fidelity, stability

    def _allocate_uid(self) -> int:
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def allocate_uid(self) -> int:
        return self._allocate_uid()
