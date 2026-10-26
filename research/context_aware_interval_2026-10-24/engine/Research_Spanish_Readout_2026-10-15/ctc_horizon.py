"""Exact CTC termination when the common search cap exceeds CTC time support."""
import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from horizon_search import HorizonBeamSearch


class TerminalCTCPrefixScoreTH(CTCPrefixScoreTH):
    """Preserve native scoring below T and score EOS exactly at the boundary."""

    def __call__(self, y, state, scoring_ids=None, att_w=None):
        output_length = len(y[0]) - 1
        if output_length < self.input_length:
            return super().__call__(y, state, scoring_ids, att_w)
        if state is None or att_w is not None or self.margin:
            raise ValueError("Terminal scoring requires a populated non-windowed CTC state")
        previous, previous_scores, minimum, maximum = state
        count = len(y)
        hypotheses = count // self.batch
        self.scoring_num = scoring_ids.size(-1) if scoring_ids is not None else 0
        width = self.scoring_num or self.odim
        mapping = None
        if scoring_ids is not None:
            mapping = torch.full((count, self.odim), -1, dtype=torch.long, device=self.device)
            rows = torch.arange(count, device=self.device).unsqueeze(1)
            mapping[rows, scoring_ids] = torch.arange(width, device=self.device)
        forward = torch.full((self.input_length, 2, count, width), self.logzero,
                             dtype=self.dtype, device=self.device)
        absolute = torch.full((count, self.odim), -torch.inf, dtype=self.dtype, device=self.device)
        if output_length == self.input_length:
            terminal = torch.logsumexp(previous, dim=1)
            for index in range(count):
                absolute[index, self.eos] = terminal[self.end_frames[index // hypotheses], index]
        increments = absolute - previous_scores
        return increments, (forward, absolute, minimum, maximum, mapping)


class SafeHorizonBeamSearch(HorizonBeamSearch):
    """Use common iteration caps while rejecting impossible CTC extensions."""

    def init_hyp(self, x):
        hypotheses = super().init_hyp(x)
        scorer = self.part_scorers.get("ctc")
        if scorer is None or not isinstance(scorer.impl, CTCPrefixScoreTH):
            raise TypeError("Expected native batched CTC prefix scorer")
        scorer.impl.__class__ = TerminalCTCPrefixScoreTH
        return hypotheses

    def search(self, running_hyps, x):
        candidates = super().search(running_hyps, x)
        finite = torch.nonzero(torch.isfinite(candidates.score), as_tuple=False).view(-1)
        return self._batch_select(candidates, finite)
