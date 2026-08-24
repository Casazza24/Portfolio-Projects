import pytest
from nlp.finbert_scorer import score_texts, SentimentResult


class TestScoreTexts:
    def test_returns_list_of_results(self):
        texts = ["Apple stock surges after earnings beat", "Market crashes on recession fears"]
        results = score_texts(texts)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, SentimentResult)

    def test_scores_sum_to_one(self):
        results = score_texts(["Nvidia reports record revenue"])
        r = results[0]
        total = r.positive + r.negative + r.neutral
        assert abs(total - 1.0) < 0.01

    def test_positive_text_scores_positive(self):
        results = score_texts(["Company reports record profits and raises guidance"])
        r = results[0]
        assert r.composite > 0

    def test_negative_text_scores_negative(self):
        results = score_texts(["Company announces massive layoffs and profit warning"])
        r = results[0]
        assert r.composite < 0

    def test_empty_input_returns_empty(self):
        results = score_texts([])
        assert results == []
