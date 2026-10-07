import numpy as np
from custody.graph import EvidenceState, CustodyEdge, CustodyGraph


def test_add_state_and_edge():
    g = CustodyGraph()
    s1 = EvidenceState(id="A", features=np.array([1.0, 2.0]), conflict_score=0.1, metadata={})
    s2 = EvidenceState(id="B", features=np.array([1.1, 2.1]), conflict_score=0.2, metadata={})
    g.add_state(s1)
    g.add_state(s2)
    g.add_edge(CustodyEdge(source="A", target="B", transform_type="re_encode", declared_params={}))

    assert g.states["A"] is s1
    assert g.states["B"] is s2
    assert len(g.edges) == 1


def test_paths_from_root_single_chain():
    g = CustodyGraph()
    for node_id in ("A", "B", "C"):
        g.add_state(EvidenceState(id=node_id, features=np.zeros(2), conflict_score=0.0, metadata={}))
    g.add_edge(CustodyEdge(source="A", target="B", transform_type="re_encode", declared_params={}))
    g.add_edge(CustodyEdge(source="B", target="C", transform_type="upload", declared_params={}))

    paths = g.paths_from_root("A")

    assert len(paths) == 1
    assert [e.transform_type for e in paths[0]] == ["re_encode", "upload"]


def test_paths_from_root_branching():
    g = CustodyGraph()
    for node_id in ("A", "B", "C"):
        g.add_state(EvidenceState(id=node_id, features=np.zeros(2), conflict_score=0.0, metadata={}))
    g.add_edge(CustodyEdge(source="A", target="B", transform_type="re_encode", declared_params={}))
    g.add_edge(CustodyEdge(source="A", target="C", transform_type="export", declared_params={}))

    paths = g.paths_from_root("A")

    transform_paths = sorted([e.transform_type for e in path] for path in paths)
    assert transform_paths == [["export"], ["re_encode"]]
