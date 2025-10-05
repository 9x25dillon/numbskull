class MatrixProcessor:
    def available(self) -> bool:
        return False

    def semantic_state_suggest(self, prefix: str, state: str):
        return []

matrix_processor = MatrixProcessor()
