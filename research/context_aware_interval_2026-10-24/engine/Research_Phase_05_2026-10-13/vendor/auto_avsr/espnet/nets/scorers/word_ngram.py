"""Pure-Python word N-gram scorer for SentencePiece shallow fusion.

The scorer deliberately has no KenLM dependency.  It applies a reproducible
Stupid-Backoff word LM while ESPnet is expanding the token beam.  Because the
acoustic/text decoder emits SentencePiece units, a word receives its LM score
when the next word-boundary piece is proposed; the final word and ``</s>`` are
scored when EOS is proposed.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import sentencepiece
import torch

from espnet.nets.scorer_interface import BatchScorerInterface


class StupidBackoffWordNGram:
    """Compact order-N word LM with Brants-style constant backoff."""

    def __init__(self, order: int = 5, alpha: float = 0.4):
        if order < 1:
            raise ValueError("order must be positive")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        self.order = int(order)
        self.alpha = float(alpha)
        self.counts = [Counter() for _ in range(self.order + 1)]
        self.total_unigrams = 0
        self.vocab_size = 0

    def fit(self, sentences: Iterable[Sequence[str]]) -> None:
        for words in sentences:
            padded = ["<s>"] * (self.order - 1) + list(words) + ["</s>"]
            for end in range(self.order - 1, len(padded)):
                for ngram_order in range(1, self.order + 1):
                    start = end - ngram_order + 1
                    if start >= 0:
                        self.counts[ngram_order][tuple(padded[start : end + 1])] += 1
        self.total_unigrams = sum(self.counts[1].values())
        self.vocab_size = len(self.counts[1])

    def token_log_score(self, history: Sequence[str], word: str) -> float:
        backoffs = 0
        maximum_order = min(self.order, len(history) + 1)
        for ngram_order in range(maximum_order, 1, -1):
            ngram = tuple((list(history) + [word])[-ngram_order:])
            count = self.counts[ngram_order].get(ngram, 0)
            if count:
                denominator = self.counts[ngram_order - 1].get(ngram[:-1], 0)
                if denominator:
                    return backoffs * math.log(self.alpha) + math.log(count / denominator)
            backoffs += 1
        unigram = self.counts[1].get((word,), 0)
        denominator = self.total_unigrams + 0.1 * (self.vocab_size + 1)
        probability = (unigram + 0.1) / max(denominator, 1.0)
        return backoffs * math.log(self.alpha) + math.log(probability)

    def score_words(self, words: Sequence[str], include_eos: bool = True) -> float:
        history = ["<s>"] * (self.order - 1)
        score = 0.0
        for word in list(words) + (["</s>"] if include_eos else []):
            score += self.token_log_score(history, word)
            history.append(word)
        return score

    def statistics(self) -> dict:
        return {
            "order": self.order,
            "backoff_alpha": self.alpha,
            "vocabulary": self.vocab_size,
            "unique_ngrams": {
                str(ngram_order): len(self.counts[ngram_order])
                for ngram_order in range(1, self.order + 1)
            },
            "total_unigram_events": self.total_unigrams,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "StupidBackoffWordNGram":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected LM object: {type(model)}")
        return model


class SentencePieceWordNGramScorer(BatchScorerInterface):
    """Expose a word N-gram as an ESPnet full scorer.

    State is reconstructed from ``ys`` rather than cached.  This is intentional:
    ESPnet full-scorer states are prefix states and do not receive the newly
    selected token in ``select_state``.  Reconstruction keeps candidate-specific
    word boundaries exact and is negligible beside the visual/attention model.
    """

    def __init__(
        self,
        model: StupidBackoffWordNGram,
        token_list: Sequence[str],
        sentencepiece_model: str | Path,
        sos: int,
        eos: int,
    ):
        self.model = model
        self.token_list = list(token_list)
        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(sentencepiece_model))
        self.sos = int(sos)
        self.eos = int(eos)
        self.boundary_ids = [
            index
            for index, piece in enumerate(self.token_list)
            if index not in (self.sos, self.eos) and piece.startswith("▁")
        ]

    def _prefix(self, token_ids: Sequence[int]) -> tuple[list[str], list[str]]:
        completed: list[str] = []
        current_pieces: list[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in (self.sos, self.eos, 0) or not 0 <= token_id < len(self.token_list):
                continue
            piece = self.token_list[token_id]
            if piece.startswith("▁") and current_pieces:
                word = self.spm.DecodePieces(current_pieces).strip()
                if word:
                    completed.append(word)
                current_pieces = []
            current_pieces.append(piece)
        return completed, current_pieces

    def _scores(self, token_ids: Sequence[int], device, dtype) -> torch.Tensor:
        completed, current_pieces = self._prefix(token_ids)
        history = ["<s>"] * (self.model.order - 1) + completed
        current_word = self.spm.DecodePieces(current_pieces).strip() if current_pieces else ""
        completion_score = (
            self.model.token_log_score(history, current_word) if current_word else 0.0
        )
        scores = torch.zeros(len(self.token_list), device=device, dtype=dtype)
        if self.boundary_ids:
            scores[self.boundary_ids] = completion_score
        eos_history = history + ([current_word] if current_word else [])
        scores[self.eos] = completion_score + self.model.token_log_score(eos_history, "</s>")
        return scores

    def score(self, y, state, x):
        return self._scores(y.tolist(), x.device, x.dtype), None

    def batch_score(self, ys, states, xs):
        scores = [
            self._scores(y.tolist(), xs.device, xs.dtype)
            for y in ys
        ]
        return torch.stack(scores), [None] * len(scores)

