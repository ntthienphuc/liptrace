"""PAViR visual-speech routing utilities."""
"""PAViR-VSR: posterior-aligned visual-identifiability routing."""

from .artifacts import (
    ArtifactStore,
    CanonicalIDMap,
    PAViRConfig,
    canonical_lookup_key,
)
from .ctc_posterior import (
    CTCPosterior,
    batched_ctc_forward_backward,
    ctc_forward_backward,
    minimum_ctc_frames,
    pool_token_span,
)

__all__ = [
    "ArtifactStore",
    "CanonicalIDMap",
    "CTCPosterior",
    "PAViRConfig",
    "batched_ctc_forward_backward",
    "canonical_lookup_key",
    "ctc_forward_backward",
    "minimum_ctc_frames",
    "pool_token_span",
]
