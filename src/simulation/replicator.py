from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Replicator:
    uid: int
    sequence: List[int]
    replication_rate: float
    fidelity: float
    stability: float
    lineage_id: int
    lineage_name: str
    parent_uid: int | None = None
    age: int = 0

    @property
    def length(self) -> int:
        return len(self.sequence)

    def signature(self) -> Tuple[int, ...]:
        return tuple(self.sequence)
