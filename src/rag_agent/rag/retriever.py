from math import fsum, hypot


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Косинусное сходство без отдельной численной зависимости."""
    if len(vec1) != len(vec2):
        raise ValueError("Embedding dimensions do not match.")
    norm_a = hypot(*vec1)
    norm_b = hypot(*vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return fsum((a / norm_a) * (b / norm_b) for a, b in zip(vec1, vec2))
