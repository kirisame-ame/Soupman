from dataclasses import dataclass
from typing import List
import random

from .environment import SimulationConfig, Environment
from .replicator import Replicator
from . import rules


@dataclass
class StepResult:
    births: List[Replicator]
    deaths: List[int]


class SimulationEngine:
    def __init__(self, config: SimulationConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.environment = Environment(config, self.rng)
        self.step_count = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.environment = Environment(self.config, self.rng)
        self.step_count = 0

    def step(self) -> StepResult:
        births: List[Replicator] = []
        deaths: List[int] = []
        survivors: List[Replicator] = []

        for rep in list(self.environment.replicators):
            rep.age += 1
            if rules.should_decay(rep, self.config, self.rng):
                deaths.append(rep.uid)
                self.environment.resources = min(
                    self.environment.max_resources,
                    self.environment.resources + rep.length,
                )
                continue

            if rules.should_replicate(rep, self.config, self.rng):
                child = rules.attempt_replication(rep, self.environment, self.config)
                if child is not None:
                    births.append(child)

            survivors.append(rep)

        survivors.extend(births)
        self.environment.replicators = survivors

        if self.config.resource_regen_rate > 0:
            self.environment.resources = min(
                self.environment.max_resources,
                self.environment.resources + self.config.resource_regen_rate,
            )

        self.step_count += 1
        return StepResult(births=births, deaths=deaths)
