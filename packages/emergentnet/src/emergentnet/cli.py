"""``emergentnet`` command line.

Examples::

    emergentnet solve --maxcut edges.txt --method sqa        # "i j [w]" per line
    emergentnet solve --qubo problem.json --method pt        # {"Q": [[...]]} or {"terms": {"i,j": w}}
    emergentnet solve --sk 20 --method qaoa --depth 3 --seed 1
    emergentnet solve --partition 4 5 6 7 8 --method exact
    emergentnet bench --sk 16 --methods exact sa pt sqa qaoa
    emergentnet memory ingest --store kb/ --attributes source < corpus.jsonl
    emergentnet memory query --store kb/ --k 5 "how do turbo decoders work"
    emergentnet backend
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__


def _problem(a):
    from .optim import QUBO, Ising, maxcut_from_edges, number_partitioning, sherrington_kirkpatrick
    if a.maxcut:
        edges = []
        with open(a.maxcut) as f:
            for line in f:
                p = line.split()
                if p and not p[0].startswith("#"):
                    edges.append((int(p[0]), int(p[1]), float(p[2]) if len(p) > 2 else 1.0))
        return maxcut_from_edges(edges), "maxcut"
    if a.qubo or a.ising:
        with open(a.qubo or a.ising) as f:
            spec = json.load(f)
        if a.qubo:
            if "terms" in spec:
                terms = {tuple(int(x) for x in k.split(",")): v for k, v in spec["terms"].items()}
                return QUBO.from_dict(terms, spec.get("n"), spec.get("offset", 0.0)), "qubo"
            return QUBO(np.array(spec["Q"]), spec.get("offset", 0.0)), "qubo"
        return Ising(np.array(spec["h"]), np.array(spec["J"]), spec.get("offset", 0.0)), "ising"
    if a.sk:
        return sherrington_kirkpatrick(a.sk, a.seed), "sk"
    if a.partition:
        return number_partitioning(a.partition), "partition"
    raise SystemExit("specify --maxcut, --qubo, --ising, --sk or --partition")


def _kw(a, method):
    kw = {}
    if method in ("sa", "pt", "sqa", "qaoa") and a.seed is not None:
        kw["seed"] = a.seed
    if a.sweeps and method in ("sa", "pt", "sqa"):
        kw["sweeps"] = a.sweeps
    if method == "qaoa":
        kw["depth"] = a.depth
    return kw


def cmd_solve(a) -> int:
    from .optim import solve
    prob, kind = _problem(a)
    r = solve(prob, a.method, **_kw(a, a.method if a.method != "auto" else ("exact" if prob.n <= 18 else "pt")))
    out = {"problem": kind, "n": int(prob.n), "method": r.method, "energy": r.energy,
           "x": [int(v) for v in r.x], "wall_time_s": round(r.wall_time, 4), "evaluations": r.evaluations}
    if kind == "maxcut":
        out["cut"] = -r.energy
    out.update({k: v for k, v in r.info.items() if isinstance(v, (int, float, str))})
    print(json.dumps(out))
    return 0


def cmd_bench(a) -> int:
    from .optim import solve
    prob, kind = _problem(a)
    ref = solve(prob, "exact").energy if prob.n <= 22 else None
    for m in a.methods:
        r = solve(prob, m, **_kw(a, m))
        rec = {"method": m, "energy": round(r.energy, 6), "wall_time_s": round(r.wall_time, 3)}
        if ref is not None:
            rec["gap"] = round(r.energy - ref, 6)
            rec["optimal"] = abs(r.energy - ref) < 1e-6
        print(json.dumps(rec), flush=True)
    return 0


def cmd_backend(a) -> int:
    from .accel import describe
    print(json.dumps(describe()))
    return 0


def cmd_memory(a) -> int:
    from pathlib import Path

    from .memory import HolographicStore
    path = Path(a.store)
    if a.action == "ingest":
        st = (HolographicStore.load(path, a.embedder) if (path / "manifest.json").exists()
              else HolographicStore(embedder=a.embedder, attributes=a.attributes or (), backend=a.backend))
        texts, metas = [], []
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                rec = json.loads(line)
                texts.append(rec["text"])
                metas.append(rec.get("metadata", {}))
            else:
                texts.append(line)
                metas.append({})
        stats = st.ingest(texts, batch_size=a.batch, metadata=metas)
        st.save(path)
        print(json.dumps({"added": len(texts), "total": len(st), **stats}))
        return 0
    st = HolographicStore.load(path, a.embedder, a.backend)
    where = dict(kv.split("=", 1) for kv in (a.where or []))
    hits = st.search(" ".join(a.query), k=a.k, where=where or None, mode=a.mode)
    if a.context:
        print(HolographicStore.format_context(hits))
    else:
        for h in hits:
            print(json.dumps({"id": h.id, "score": round(h.score, 4), "text": h.text, "metadata": h.metadata}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="emergentnet", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def problem_args(p):
        g = p.add_mutually_exclusive_group()
        g.add_argument("--maxcut")
        g.add_argument("--qubo")
        g.add_argument("--ising")
        g.add_argument("--sk", type=int, help="random Sherrington-Kirkpatrick instance of size N")
        g.add_argument("--partition", type=float, nargs="+")
        p.add_argument("--seed", type=int)
        p.add_argument("--sweeps", type=int)
        p.add_argument("--depth", type=int, default=3)

    p = sub.add_parser("solve")
    problem_args(p)
    p.add_argument("--method", default="auto", choices=["auto", "exact", "sa", "pt", "sqa", "qaoa"])
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("bench")
    problem_args(p)
    p.add_argument("--methods", nargs="+", default=["sa", "pt", "sqa"])
    p.set_defaults(fn=cmd_bench)

    sub.add_parser("backend", help="show the selected array backend / GPU").set_defaults(fn=cmd_backend)

    p = sub.add_parser("memory", help="holographic RAG store")
    p.add_argument("action", choices=["ingest", "query"])
    p.add_argument("query", nargs="*")
    p.add_argument("--store", required=True)
    p.add_argument("--embedder", default="hashing:1024")
    p.add_argument("--backend", default="auto")
    p.add_argument("--attributes", nargs="*")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--where", nargs="*", help="attr=value")
    p.add_argument("--mode", choices=["soft", "hard"], default="hard")
    p.add_argument("--context", action="store_true", help="print an LLM-ready context block")
    p.set_defaults(fn=cmd_memory)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
