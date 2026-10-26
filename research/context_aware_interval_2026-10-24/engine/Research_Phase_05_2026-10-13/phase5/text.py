"""Vocabulary-faithful tokenizer and transparent scoring units."""

import unicodedata
import sentencepiece as spm


def normalize(text, punctuation=False):
    text = unicodedata.normalize('NFC', str(text)).lower()
    if punctuation:
        text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    return ' '.join(text.split())


class Tokenizer:
    def __init__(self, model, units, language):
        self.processor = spm.SentencePieceProcessor(model_file=str(model))
        entries = {}
        for line in units.read_text(encoding='utf-8').splitlines():
            token, identifier = line.rsplit(maxsplit=1)
            entries[int(identifier)] = token
        if set(entries) != set(range(1, len(entries) + 1)):
            raise ValueError('Vocabulary IDs must be contiguous, starting at one')
        self.tokens = ['<blank>'] + [entries[i] for i in sorted(entries)] + ['<eos>']
        self.mapping = {token: index for index, token in enumerate(self.tokens)}
        pieces = {self.processor.id_to_piece(i) for i in range(self.processor.get_piece_size())
                  if not self.processor.is_control(i)}
        if not pieces.issubset(self.mapping):
            raise ValueError('SentencePiece pieces missing from checkpoint vocabulary')
        self.language = language

    def encode(self, text):
        value = reference_text(text, self.language)
        if self.language != 'vi':
            value = value.upper()
        # Unknown string pieces may contain the input surface (e.g. "846"), not "<unk>".
        # Canonical SentencePiece IDs retain the checkpoint's existing unknown unit.
        pieces = self.processor.encode(value, out_type=int)
        ids = [self.mapping[self.processor.id_to_piece(p)] for p in pieces]
        if not ids or any(i <= 0 or i >= len(self.tokens) - 1 for i in ids):
            raise ValueError('Empty or invalid token target')
        return ids

    def decode(self, ids):
        return ''.join(self.tokens[int(i)] for i in ids if 0 < int(i) < len(self.tokens) - 1).replace(
            '\u2581', ' ').replace('<space>', ' ').strip()


def required_frames(ids):
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def reference_text(text, language):
    value = normalize(text)
    if language == 'vi':
        value = ''.join(c if unicodedata.category(c)[0] in 'LNM' else ' ' for c in value)
        value = ' '.join(value.split())
    return value


def edit_counts(reference, hypothesis):
    # Each tuple is distance, substitutions, deletions, insertions.
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            d, s, de, ins = previous[j - 1]
            diagonal = (d + (ref != hyp), s + (ref != hyp), de, ins)
            d, s, de, ins = previous[j]
            deletion = (d + 1, s, de + 1, ins)
            d, s, de, ins = current[j - 1]
            insertion = (d + 1, s, de, ins + 1)
            current.append(min([diagonal, deletion, insertion], key=lambda x: x[0]))
        previous = current
    _, s, d, i = previous[-1]
    return {'substitutions': s, 'deletions': d, 'insertions': i,
            'errors': s + d + i, 'units': len(reference)}


def score(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    word = edit_counts(ref.split(), hyp.split())
    character = edit_counts(list(ref.replace(' ', '')), list(hyp.replace(' ', '')))
    secondary = edit_counts(normalize(ref, True).split(), normalize(hyp, True).split())
    return {'word': word, 'character': character, 'no_punctuation': secondary}
