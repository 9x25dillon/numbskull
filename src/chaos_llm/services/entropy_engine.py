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

entropy_engine = EntropyEngine()
