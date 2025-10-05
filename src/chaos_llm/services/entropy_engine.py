cursor/bc-f408c7bd-bc2a-48a4-bc8d-0989f628ad52-ef2e
class EntropyEngine:
    def score_token(self, token: str) -> float:
        s = token or ""
        if not s:
            return 0.0
        import math
        counts = {}
        for c in s:
            counts[c] = counts.get(c, 0) + 1
        n = len(s)
        ent = 0.0
        for v in counts.values():
            p = v / n
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    def get_volatility_signal(self, token: str):
        return None

from __future__ import annotations

class EntropyEngine:
    def score_token(self, token_text: str) -> float:
        if not token_text:
            return 0.0
        # Simple normalized entropy proxy: unique chars / length
        unique = len(set(token_text))
        return unique / max(1, len(token_text))

    def get_volatility_signal(self, token_text: str) -> float:
        # Heuristic volatility: presence of punctuation/operators
        ops = sum(1 for c in token_text if c in "()[]{}+-/*=,<>&|!?")
        return ops / max(1, len(token_text))


entropy_engine = EntropyEngine()
