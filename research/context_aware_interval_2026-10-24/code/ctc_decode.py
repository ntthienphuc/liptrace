"""Pure CTC prefix beam, with explicit frame-token and prefix pruning.

Scores sum original CTC path mass without renormalizing forbidden EOS away.
EOS paths are excluded altogether, never merged into blank or used to stop time.
All frames contribute. Exact path summation requires token_prune=None AND a beam
wide enough to retain every feasible prefix; finite beam width is approximate.
"""

import math
from numbers import Integral

import numpy as np


NEGATIVE_INFINITY = float("-inf")


def _logadd(left, right):
    if left == NEGATIVE_INFINITY:
        return right
    if right == NEGATIVE_INFINITY:
        return left
    if left < right:
        left, right = right, left
    return left + math.log1p(math.exp(right - left))


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _scores(logp):
    values = np.asarray(logp, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("logp must have shape [frames, vocabulary] with nonempty vocabulary")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("logp cannot contain NaN or positive infinity")
    if np.any(values > 1e-6):
        raise ValueError("logp must contain log probabilities, not positive logits")
    return values


def prefix_beam(logp, beam_size=40, blank=0, eos=None, token_prune=64):
    """Decode without attention, language model, reference or EOS lexical output.

    Per frame, pruning keeps the top K permitted vocabulary IDs, blank, and every
    surviving prefix's last token. Ties use token ID then lexicographic prefix.
    ``None`` disables token pruning but not the separately declared beam pruning.
    Empty time returns the empty prefix with log score zero. Impossible allowed
    paths raise ValueError so durable JSON never silently receives negative infinity.
    """
    values = _scores(logp)
    beam_size = _integer(beam_size, "beam_size", 1)
    blank = _integer(blank, "blank")
    vocabulary_size = values.shape[1]
    if blank != 0:
        raise ValueError("Confirmation models require blank vocabulary ID 0")
    if eos is not None:
        eos = _integer(eos, "eos")
        if eos >= vocabulary_size or eos == blank:
            raise ValueError("eos must be a vocabulary ID distinct from blank")
    if token_prune is not None:
        token_prune = _integer(token_prune, "token_prune", 1)
    permitted = np.array([token for token in range(vocabulary_size) if token != eos], dtype=np.int64)
    beam = {(): (0.0, NEGATIVE_INFINITY)}
    for frame_index, frame in enumerate(values):
        candidates = permitted
        if token_prune is not None and token_prune < len(permitted):
            probabilities = frame[permitted]
            threshold = np.partition(probabilities, len(permitted) - token_prune)[-token_prune]
            higher = permitted[probabilities > threshold]
            ties = permitted[probabilities == threshold]
            candidates = np.concatenate((higher, ties[:token_prune - len(higher)]))
        symbols = set(int(token) for token in candidates)
        symbols.add(blank)
        symbols.update(prefix[-1] for prefix in beam if prefix)
        symbols.discard(blank)
        lexical = sorted(symbols)
        updated = {}

        def accumulate(prefix, component, score):
            if score == NEGATIVE_INFINITY:
                return
            masses = updated.setdefault(prefix, [NEGATIVE_INFINITY, NEGATIVE_INFINITY])
            masses[component] = _logadd(masses[component], score)

        for prefix, (blank_mass, lexical_mass) in beam.items():
            total = _logadd(blank_mass, lexical_mass)
            accumulate(prefix, 0, total + float(frame[blank]))
            for token in lexical:
                probability = float(frame[token])
                if probability == NEGATIVE_INFINITY:
                    continue
                if prefix and token == prefix[-1]:
                    accumulate(prefix, 1, lexical_mass + probability)
                    accumulate(prefix + (token,), 1, blank_mass + probability)
                else:
                    accumulate(prefix + (token,), 1, total + probability)
        if not updated:
            raise ValueError(f"No feasible CTC path at frame {frame_index} after EOS exclusion/pruning")
        ranked = sorted(updated.items(), key=lambda item: (-_logadd(*item[1]), item[0]))
        beam = dict(ranked[:beam_size])
    prefix, masses = min(beam.items(), key=lambda item: (-_logadd(*item[1]), item[0]))
    return {
        "tokens": list(prefix),
        "score": float(_logadd(*masses)),
        "beam_size": beam_size,
        "token_prune": token_prune,
        "algorithm": "ctc_prefix_beam",
    }
