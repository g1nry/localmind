import pytest

from rag_agent.rag.retriever import cosine_similarity


@pytest.mark.parametrize(
    "left,right,expected",
    [([3, 4], [6, 8], 1), ([3, 4], [-3, -4], -1), ([0, 0], [3, 4], 0)],
)
def test_cosine_similarity_is_normalized(left, right, expected):
    assert cosine_similarity(left, right) == pytest.approx(expected)


def test_different_embedding_dimensions_are_rejected():
    with pytest.raises(ValueError):
        cosine_similarity([1, 0], [1])
