"""Data Processor for extracting span intervals from marker-paired sequences.

This module defines a processor that scans a sequence for specific begin and end markers
and extracts span intervals corresponding to those marker pairs. It is useful in tasks
such as extracting annotated segments (e.g., entities or labeled text spans) from
token sequences or label sequences in NLP workflows.
"""

from hyped.typing import Sequence, Int, String, Bool
from hyped.core import BaseDataProcessor, BaseDataProcessorConfig, process_mode, RunContext
import numpy as np

from hyped.extensions.nlp.nodes.spans.utils import Spans
from typing import TypeVar


class ExtractSpansConfig(BaseDataProcessorConfig):
    """Configuration for ExtractSpans processor."""
    
    allow_unclosed_initial_span: bool = False
    """Allow unclosed initial span.
    
    If True, allows a leading unmatched end marker to produce a span that starts at the
    beginning of the sequence.
    """

    allow_unclosed_final_span: bool = False
    """Allow unclosed final span.
    
    If True, allows a final unmatched begin marker to produce a span that extends to the
    end of the sequence.
    """

    ignore_unmatched_markers: bool = False


class ExtractSpans(BaseDataProcessor[ExtractSpansConfig]):
    """Processor to extract span intervals based on paired markers.

    This processor scans through a sequence and extracts spans that are bounded by
    matching begin and end markers. It supports optionally allowing a final unmatched
    begin or a leading unmatched end marker to create open-ended spans.
    """

    T = TypeVar("T", Int, String)

    @process_mode(batched=False, backend="python")
    def process(
        self,
        ctx: RunContext,
        sequence: Sequence[T],
        begin_marker: T,
        end_marker: T
    ) -> Spans:
        """Extracts span intervals from a sequence based on begin and end markers.

        Args:
            ctx (RunContext): The runtime context for the data processor.
            sequence (Sequence[T]): A sequence of elements within which spans are identified.
            begin_marker (T): The marker that denotes the start of a span.
            end_marker (T): The marker that denotes the end of a span.

        Returns:
            Spans: A list of (start, end) index pairs representing extracted spans.

        Raises:
            ValueError: If the number of begin and end markers do not match and neither
                        open-ended config flag is set appropriately.
            ValueError: If any begin marker appears after its corresponding end marker.
        """
        sequence = np.asarray(sequence)
        begin_indices = np.where(sequence == begin_marker)[0]
        end_indices = np.where(sequence == end_marker)[0] + 1

        # Handle unmatched end marker at the beginning
        if self.config.allow_unclosed_initial_span and (
            (
                (len(begin_indices) != 0)
                and (len(end_indices) != 0)
                and (begin_indices[0] > end_indices[0])
            ) or (
                (len(begin_indices) == 0)
                and (len(end_indices) != 0)
            )
        ):
            begin_indices = np.insert(begin_indices, 0, 0)

        # Handle unmatched begin marker at the end
        if self.config.allow_unclosed_final_span and (
            (
                (len(begin_indices) != 0)
                and (len(end_indices) != 0)
                and (begin_indices[-1] > end_indices[-1])
            ) or (
                (len(begin_indices) != 0)
                and (len(end_indices) == 0)
            )
        ):
            end_indices = np.append(end_indices, len(sequence))

        if self.config.ignore_unmatched_markers:
            # TODO: tests for this feature
            M = (begin_indices[:, None] < end_indices[None, :])
            M[:-1, :] = ~(M[:-1, :] == M[1:, :]).all(axis=1, keepdims=True)
            M[:, 1:] = ~(M[:, :-1] == M[:, 1:]).all(axis=0, keepdims=True)
            begin_indices = begin_indices[M.any(axis=1)]
            end_indices = end_indices[M.any(axis=0)]
            assert len(begin_indices) == len(end_indices)
            assert (begin_indices < end_indices).all()

        # Disallow any other mismatch
        if len(begin_indices) != len(end_indices):
            raise ValueError("Mismatched number of begin and end markers")

        # Ensure ordering: begin must always precede end
        if not np.all(begin_indices < end_indices):
            raise ValueError("Begin markers must precede end markers in order")

        return list(zip(begin_indices, end_indices))
