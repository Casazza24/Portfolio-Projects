from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from config.sentiment import FINBERT_MODEL_NAME, FINBERT_BATCH_SIZE, FINBERT_MAX_LENGTH


@dataclass
class SentimentResult:
    positive: float
    negative: float
    neutral: float
    composite: float  # P(positive) - P(negative)


_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def score_texts(texts: list[str]) -> list[SentimentResult]:
    if not texts:
        return []

    model, tokenizer = _load_model()
    results = []

    for i in range(0, len(texts), FINBERT_BATCH_SIZE):
        batch = texts[i : i + FINBERT_BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

        # FinBERT label order: positive=0, negative=1, neutral=2
        for row in probs:
            pos, neg, neu = float(row[0]), float(row[1]), float(row[2])
            results.append(SentimentResult(
                positive=pos, negative=neg, neutral=neu,
                composite=pos - neg,
            ))

    return results
