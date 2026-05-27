import random
from typing import List, Tuple

from .environment import (
    SimulationConfig,
    Environment,
    clamp,
    enforce_trait_total,
    apply_stability_cap,
    apply_stability_replication_tradeoff,
)
from .replicator import Replicator


def should_replicate(
    rep: Replicator, config: SimulationConfig, rng: random.Random
) -> bool:
    rate = clamp(rep.replication_rate * config.replication_multiplier)
    return rng.random() < rate


def should_decay(rep: Replicator, config: SimulationConfig, rng: random.Random) -> bool:
    survival = clamp(rep.stability * config.stability_multiplier)
    return rng.random() > survival


def mutate_sequence(
    sequence: List[int],
    fidelity: float,
    config: SimulationConfig,
    rng: random.Random,
) -> Tuple[List[int], bool]:
    error_rate = clamp((1.0 - fidelity) * config.mutation_multiplier)
    mutated = False
    new_sequence: List[int] = []
    for value in sequence:
        if rng.random() < error_rate:
            mutated = True
            if config.alphabet_size > 1:
                new_value = rng.randrange(config.alphabet_size - 1)
                if new_value >= value:
                    new_value += 1
                new_sequence.append(new_value)
            else:
                new_sequence.append(value)
        else:
            new_sequence.append(value)
    return new_sequence, mutated


def mutate_traits(
    replication_rate: float,
    fidelity: float,
    stability: float,
    config: SimulationConfig,
    rng: random.Random,
) -> Tuple[float, float, float]:
    if rng.random() >= config.trait_mutation_rate:
        return replication_rate, fidelity, stability

    delta = lambda: rng.uniform(
        -config.trait_mutation_strength, config.trait_mutation_strength
    )
    replication_rate = clamp(replication_rate + delta())
    fidelity = clamp(fidelity + delta())
    stability = clamp(stability + delta())
    replication_rate, fidelity, stability = enforce_trait_total(
        replication_rate,
        fidelity,
        stability,
        config.trait_total_cap,
    )
    stability = apply_stability_cap(stability, config.stability_cap)
    replication_rate = apply_stability_replication_tradeoff(
        replication_rate,
        stability,
        config.stability_replication_tradeoff_floor,
        config.stability_replication_tradeoff_strength,
    )
    return replication_rate, fidelity, stability


def attempt_replication(
    rep: Replicator, env: Environment, config: SimulationConfig
) -> Replicator | None:
    cost = rep.length
    if env.resources < cost:
        return None

    new_sequence, _ = mutate_sequence(rep.sequence, rep.fidelity, config, env.rng)
    replication_rate, fidelity, stability = mutate_traits(
        rep.replication_rate, rep.fidelity, rep.stability, config, env.rng
    )
    child = Replicator(
        uid=env.allocate_uid(),
        sequence=new_sequence,
        replication_rate=replication_rate,
        fidelity=fidelity,
        stability=stability,
        lineage_id=rep.lineage_id,
        lineage_name=rep.lineage_name,
        parent_uid=rep.uid,
    )
    env.resources -= cost
    return child
