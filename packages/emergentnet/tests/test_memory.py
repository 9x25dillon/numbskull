import numpy as np
import pytest

from emergentnet.accel import Backend, get_backend
from emergentnet.memory import (AssociativeTrace, HashingEmbedder, HolographicStore, bind, capacity, symbol,
                                unbind, unitary)

DOCS = [
    ("Reed-Solomon codes correct burst errors over GF(256)", "fec", "alice"),
    ("LDPC decoders use belief propagation on sparse graphs", "fec", "bob"),
    ("Simulated annealing lowers temperature to find ground states", "optim", "alice"),
    ("QAOA alternates cost and mixer unitaries", "optim", "carol"),
    ("OFDM uses a cyclic prefix to absorb multipath", "dsp", "bob"),
    ("Turbo codes iterate two BCJR decoders", "fec", "carol"),
]


def make_store(backend="numpy"):
    st = HolographicStore(embedder="hashing:1024", attributes=("topic", "author"), backend=backend)
    st.add([d[0] for d in DOCS], metadata=[{"topic": d[1], "author": d[2]} for d in DOCS])
    return st


def test_hrr_algebra():
    a, b = unitary(512, 1), unitary(512, 2)
    assert np.allclose(unbind(bind(a, b), a), b, atol=1e-5)
    assert abs(float(a @ b)) < 5 / np.sqrt(512)
    assert np.array_equal(symbol("x", 256, "ns"), symbol("x", 256, "ns"))


def test_trace_recall_within_capacity():
    d = 1024
    t = AssociativeTrace(d)
    pairs = [(f"k{i}", f"v{i % 32}") for i in range(capacity(d, 32))]
    for k, v in pairs:
        t.store(k, v)
    assert np.mean([t.recall(k)[0][0] == v for k, v in pairs]) == 1.0


def test_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(256)
    a, b = e.encode(["hello world"]), e.encode(["hello world"])
    assert np.array_equal(a, b) and np.isclose(np.linalg.norm(a), 1.0)


def test_lexical_retrieval():
    st = make_store()
    top = st.search("belief propagation decoders", k=1)[0]
    assert top.id == "1"


def test_soft_attribute_boost_and_hard_filter():
    st = make_store()
    # soft = additive boost: carol's decoder doc overtakes bob's equally relevant one
    plain = st.search("decoders", k=6)
    soft = st.search("decoders", k=6, where={"author": "carol"})
    rank = lambda hits, i: [h.id for h in hits].index(i)
    assert rank(soft, "5") < rank(soft, "1") and rank(soft, "3") < rank(plain, "3")
    # a large weight makes the boost dominate content similarity (filter-like)
    strong = st.search("decoders", k=2, where={"author": "carol"}, attr_weight=1.0)
    assert all(h.metadata["author"] == "carol" for h in strong)
    hard = st.search("decoders", k=5, where={"author": "bob"}, mode="hard")
    assert {h.metadata["author"] for h in hard} == {"bob"} and len(hard) == 2


def test_attribute_recovery_from_vector_only():
    st = make_store()
    for i, d in enumerate(DOCS):
        assert st.recover_attribute(str(i), "author")[0][0] == d[2]
        assert st.recover_attribute(str(i), "topic")[0][0] == d[1]


def test_persistence_roundtrip(tmp_path):
    st = make_store()
    st.save(tmp_path / "kb")
    st.save(tmp_path / "kb")  # overwrite path is atomic
    st2 = HolographicStore.load(tmp_path / "kb")
    assert len(st2) == len(st)
    q = "cyclic prefix multipath"
    assert [h.id for h in st2.search(q, k=3)] == [h.id for h in st.search(q, k=3)]
    assert st2.recover_attribute("4", "author")[0][0] == "bob"


def test_delete_upsert_and_batch_queries():
    st = make_store()
    assert st.delete(["1", "missing"]) == 1 and len(st) == 5
    st.upsert(["LDPC min-sum decoding"], ids=["1"], metadata=[{"topic": "fec", "author": "dave"}])
    assert st.get("1").metadata["author"] == "dave"
    res = st.search(["turbo BCJR", "annealing temperature"], k=1)
    assert [r[0].id for r in res] == ["5", "2"]
    with pytest.raises(ValueError):
        st.add(["dup"], ids=["1"])


def test_chunked_topk_matches_bruteforce():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5000, 64)).astype(np.float32)
    Q = rng.normal(size=(3, 64)).astype(np.float32)
    b = Backend("numpy")
    s, i = b.topk(b.asarray(M), Q, 7, chunk=777)
    ref = np.argsort(-(Q @ M.T), axis=1)[:, :7]
    assert np.array_equal(i, ref)


def test_torch_backend_matches_numpy():
    torch = pytest.importorskip("torch")
    del torch
    tb = get_backend("torch")
    st_np, st_t = make_store("numpy"), make_store("torch")
    assert tb.name == "torch"
    for q in ("decoders", "annealing"):
        a, b = st_np.search(q, k=6), st_t.search(q, k=6)
        assert a[0].id == b[0].id  # lower ranks contain exact ties whose order is backend-defined
        assert np.allclose([h.score for h in a], [h.score for h in b], atol=1e-5)


def test_ingest_stream_large():
    st = HolographicStore(embedder=HashingEmbedder(256), backend="numpy")
    stats = st.ingest((f"document number {i} about topic {i % 17}" for i in range(3000)), batch_size=500)
    assert len(st) == 3000 and stats["texts"] == 3000
    assert st.search("document number 1234 about topic 10", k=1)[0].id == "1234"
