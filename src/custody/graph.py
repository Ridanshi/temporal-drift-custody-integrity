from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvidenceState:
    id: str
    features: np.ndarray
    conflict_score: float
    metadata: dict


@dataclass
class CustodyEdge:
    source: str
    target: str
    transform_type: str
    declared_params: dict


@dataclass
class CustodyGraph:
    states: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    _outgoing: dict = field(default_factory=dict)

    def add_state(self, state: EvidenceState) -> None:
        self.states[state.id] = state
        self._outgoing.setdefault(state.id, [])

    def add_edge(self, edge: CustodyEdge) -> None:
        self.edges.append(edge)
        self._outgoing.setdefault(edge.source, []).append(edge)

    def paths_from_root(self, root_id: str) -> list[list[CustodyEdge]]:
        def walk(node_id: str) -> list[list[CustodyEdge]]:
            outgoing = self._outgoing.get(node_id, [])
            if not outgoing:
                return [[]]
            paths = []
            for edge in outgoing:
                for rest in walk(edge.target):
                    paths.append([edge] + rest)
            return paths

        return walk(root_id)
