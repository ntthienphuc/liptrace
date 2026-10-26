"""Reference-free CTC alignment and predicted-word projection; NumPy only."""
import numpy as np


def collapse_path(path, blank=0, eos=None):
    tokens, spans = [], []
    previous = None
    for frame, label in enumerate(path):
        label = int(label)
        if label != previous and label not in (blank, eos):
            tokens.append(label)
            spans.append([frame, frame + 1])
        elif label == previous and label not in (blank, eos):
            spans[-1][1] = frame + 1
        previous = label
    return tokens, spans


def align_prediction(logp, tokens, blank=0):
    """Viterbi CTC forced alignment of a PREDICTED token sequence, never a reference."""
    logp = np.asarray(logp, dtype=np.float64)
    frames, vocabulary = logp.shape
    if frames == 0 or any(token <= blank or token >= vocabulary for token in tokens):
        raise ValueError("Invalid logits or target token IDs")
    if not tokens:
        return {"spans": [], "score": float(logp[:, blank].sum())}
    required = len(tokens) + sum(left == right for left, right in zip(tokens, tokens[1:]))
    if required > frames:
        return None
    labels = np.full(2 * len(tokens) + 1, blank, dtype=np.int64)
    labels[1::2] = tokens
    states = len(labels)
    previous = np.full(states, -np.inf)
    previous[0] = logp[0, blank]
    previous[1] = logp[0, tokens[0]]
    parents = np.full((frames, states), -1, dtype=np.int64)
    skip = np.zeros(states, dtype=bool)
    skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    indices = np.arange(states)
    for frame in range(1, frames):
        advance = np.concatenate(([-np.inf], previous[:-1]))
        jump = np.concatenate(([-np.inf, -np.inf], previous[:-2]))
        jump[~skip] = -np.inf
        candidates = np.stack([previous, advance, jump])
        choice = candidates.argmax(axis=0)
        previous = candidates[choice, indices] + logp[frame, labels]
        parents[frame] = indices - choice
    state = states - 1 if previous[-1] >= previous[-2] else states - 2
    score = float(previous[state])
    if not np.isfinite(score):
        return None
    route = np.empty(frames, dtype=np.int64)
    for frame in range(frames - 1, -1, -1):
        route[frame] = state
        if frame:
            state = int(parents[frame, state])
    spans = []
    for index in range(len(tokens)):
        occupied = np.flatnonzero(route == 2 * index + 1)
        if not len(occupied):
            raise AssertionError("Feasible alignment omitted a token")
        spans.append([int(occupied[0]), int(occupied[-1]) + 1])
    return {"spans": spans, "score": score}


def word_spans(tokens, spans, vocabulary):
    """Match Tokenizer.decode whitespace and reconstruct each predicted word."""
    if len(tokens) != len(spans):
        raise ValueError("Token and span counts differ")
    words = []
    letters = []
    positions = []

    def flush():
        if letters:
            words.append({
                "text": "".join(letters),
                "start": min(spans[position][0] for position in positions),
                "end": max(spans[position][1] for position in positions),
                "token_positions": sorted(set(positions)),
            })
            letters.clear()
            positions.clear()

    for position, token in enumerate(tokens):
        piece = vocabulary[token].replace("\u2581", " ").replace("<space>", " ")
        for letter in piece:
            if letter.isspace():
                flush()
            else:
                letters.append(letter)
                positions.append(position)
    flush()
    return words


def project_words(tokens, spans, vocabulary, start, end):
    words = word_spans(tokens, spans, vocabulary)
    selected = [word for word in words if word["end"] > start and word["start"] < end]
    return " ".join(word["text"] for word in selected), words


def project_tokens(tokens, spans, start, end):
    return [token for token, (lower, upper) in zip(tokens, spans) if upper > start and lower < end]
