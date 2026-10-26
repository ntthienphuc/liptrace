"""Explicit iteration cap with native maxlenratio=0 end detection retained."""
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.e2e_asr_common import end_detect


class HorizonBeamSearch(BatchBeamSearch):
    """Reuse native initialization, scoring and post_process without vendor edits."""

    def forward(self, x, maxlenratio=0.0, minlenratio=0.0, *, step_cap=None):
        if maxlenratio != 0.0 or minlenratio != 0.0:
            raise ValueError("This controlled comparison requires both native ratios to be zero")
        cap = int(x.shape[0]) if step_cap is None else step_cap
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError("step_cap must be a positive integer")
        running = self.init_hyp(x)
        ended = []
        forced_ids = set()
        diagnostic = {
            "input_states": int(x.shape[0]), "step_cap": cap,
            "maxlenratio": 0.0, "end_detection_enabled": True,
            "iterations": 0, "end_detection_calls": 0,
            "forced_eos_hypotheses": 0, "cap_reached": False,
            "stop_reason": None,
        }
        for iteration in range(cap):
            best = self.search(running, x)
            ended_before = len(ended)
            running = self.post_process(iteration, cap, 0.0, best, ended)
            diagnostic["iterations"] = iteration + 1
            if iteration == cap - 1:
                diagnostic["cap_reached"] = True
                forced_ids.update(id(hypothesis) for hypothesis in ended[ended_before:])
            diagnostic["end_detection_calls"] += 1
            detected = end_detect([hypothesis.asdict() for hypothesis in ended], iteration)
            if detected:
                diagnostic["stop_reason"] = "end_detect"
                break
            if len(running) == 0:
                diagnostic["stop_reason"] = "cap_forced_eos" if iteration == cap - 1 else "no_running"
                break
        if diagnostic["stop_reason"] is None:
            diagnostic["stop_reason"] = "cap_exhausted"
        result = sorted(ended, key=lambda hypothesis: hypothesis.score, reverse=True)
        diagnostic["forced_eos_hypotheses"] = len(forced_ids)
        diagnostic["best_forced_eos"] = bool(result and id(result[0]) in forced_ids)
        diagnostic["ended_hypotheses"] = len(result)
        self.last_diagnostic = diagnostic
        return result
