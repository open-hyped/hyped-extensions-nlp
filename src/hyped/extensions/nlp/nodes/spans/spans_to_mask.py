"""Data Processor for converting span annotations into binary masks.

This module defines a processor that takes a sequence of spans and a total sequence length,
and returns a boolean mask indicating which positions fall inside any of the input spans.
"""

import numpy as np
from hyped.typing import Sequence, Int, Bool
from hyped.core import BaseDataProcessor, BaseDataProcessorConfig, process_mode, RunContext
from hyped.extensions.nlp.nodes.spans.utils import Spans


class SpansToMaskConfig(BaseDataProcessorConfig):
    """Configuration for SpansToMask processor."""


class SpansToMask(BaseDataProcessor[SpansToMaskConfig]):
    """Processor to convert span annotations into a boolean mask.

    This processor takes in a list of span boundaries and the total sequence length,
    and returns a boolean mask where each position is set to :code:`True` if it is
    covered by at least one span, and :code:`False` otherwise.
    """

    @process_mode(batched=False, backend="python")
    def process(self, ctx: RunContext, spans: Spans, length: Int) -> Sequence[Bool]:
        """Generates a boolean mask from input spans over a sequence of given length.

        Args:
            ctx (RunContext): The runtime context for the data processor, providing
                execution-time metadata and utilities.
            spans (Spans): A sequence of spans, where each span is defined as a pair
                of integer start and end indices, e.g., :code:`(start, end)`. These
                spans may overlap or be disjoint.
            length (Int): The total length of the sequence over which to generate the mask.

        Returns:
            Sequence[Bool]: A boolean sequence of the same length as :code:`length`,
            where each position is :code:`True` if it is included in at least one span,
            and :code:`False` otherwise.
        """
        spans = np.array(spans)
        mask = np.zeros(length, dtype=int)

        print(spans)
        print(mask, length)

        if spans.ndim == 2:
            # Increment at span starts, decrement at span ends
            np.add.at(mask, spans[:, 0], 1)
            np.add.at(mask, spans[:, 1], -1)

        # Compute the cumulative sum and threshold to generate the mask
        return np.cumsum(mask) > 0
