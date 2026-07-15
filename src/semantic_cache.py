import numpy as np

class SemanticCache:
    def __init__(self, embedder, threshold: float):
        self.embedder = embedder          # REUSE existing MiniLM — passed in, not created here
        self.threshold = threshold        # tuned via the calibration set
        self.keys: list[np.ndarray] = []  # stored query embeddings
        self.answers: list[str] = []      # list of cached answers
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return float(dot / (norm_a * norm_b))

    def get(self, query: str):
        # embed the query, compare to keys, return answer if above threshold
        query_embedding = self.embedder.encode(query)
        max_similarity = -1.0
        best_answer = None  # empty array to hold the best answer
        for i in range(len(self.keys)):
            similarity = self._cosine(query_embedding, self.keys[i])
            if similarity >= self.threshold:
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_answer = self.answers[i]
        if max_similarity >= self.threshold:
            self.hits += 1
            return best_answer
            
        self.misses += 1
        return None

    def put(self, query: str, answer: str):
        # embed the query, store the embedding and answer
        query_embedding = self.embedder.encode(query)
        self.keys.append(query_embedding)
        self.answers.append(answer)


    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }