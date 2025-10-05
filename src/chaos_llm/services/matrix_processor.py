from __future__ import annotations
from typing import List

class MatrixProcessor:
    def available(self) -> bool:
        # Stub off by default; set True if you wire a real vector index
        return False

    def semantic_state_suggest(self, prefix: str, state: str) -> List[str]:
        # Simple placeholder: n-gram expansions
        base = (prefix or "").upper()
        if not base:
            return ["SELECT", "FILTER", "GROUP", "ORDER"]
        return [base + s for s in ["_A", "_B", "_C"]]

matrix_processor = MatrixProcessor()
