class MotifEngine:
    def detect_tags(self, token: str):
        token = (token or "").lower()
        tags = []
        for kw in ("sum", "mean", "var", "diff", "simplify"):
            if kw in token:
                tags.append(f"{kw.upper()}_HINT")
        return tags

motif_engine = MotifEngine()
