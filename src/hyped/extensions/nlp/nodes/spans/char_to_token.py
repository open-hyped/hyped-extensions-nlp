"""Data Processor for converting character spans to token spans.

This module defines the functionality required to process character spans and convert them into
token spans, which are useful for various Natural Language Processing (NLP) tasks such as Named
Entity Recognition (NER).
"""
import numpy as np
from typing_extensions import Annotated

from hyped.core import BaseDataProcessor, BaseDataProcessorConfig, RunContext
from hyped.extensions.nlp.nodes.spans.utils import compute_spans_overlap_matrix
from hyped.typing import Int, Len, Sequence


class CharToTokenSpansConfig(BaseDataProcessorConfig):
    """Configuration for ChrToTokSpans."""

    include_partial_start: bool = True
    """Whether to include tokens that are only partly covered at the beginning of the query span."""

    include_partial_end: bool = True
    """Whether to include tokens that are only partly covered at the end of the query span."""


Span = Annotated[Sequence[Int], Len(2)]
Spans = Sequence[Span]
ChrSpansLength = Len()
QuerySpansLength = Len()


class CharToTokenSpans(BaseDataProcessor[CharToTokenSpansConfig]):
    """Processor to convert character spans to token spans.

    This processor computes the span overlap matrix between query spans and character spans,
    and then converts the overlapping spans into token spans.
    """

    def process(
        self,
        ctx: RunContext,
        chr_spans: Annotated[Spans, ChrSpansLength],
        query_spans: Annotated[Spans, QuerySpansLength],
        special_tokens_mask: Annotated[Sequence[Int], ChrSpansLength] | None = None,
    ) -> Annotated[Spans, QuerySpansLength]:
        """Converts character spans into token spans based on their overlap with query spans.

        Args:
            ctx (RunContext): The runtime context for the data processor, providing execution
                environment details.
            chr_spans (Annotated[Spans, ChrSpansLength]): A sequence of character spans
                (length :code:`ChrSpansLength`) of the tokens, represented as a sequence of
                :code:`[start, end]` positions.
            query_spans (Annotated[Spans, QuerySpansLength]): A sequence of query spans
                (length :code:`QuerySpansLength`), represented as a
                sequence of :code:`[start, end]` positions.
            special_tokens_mask (Annotated[Sequence[Int], ChrSpansLength] | None): A mask indicating
                special tokens. Tokens marked as special are ignored when computing overlaps.
                Defaults to :code:`None`.

        Returns:
            Annotated[Spans, QuerySpansLength]: A sequence of token spans aligned with the input
                query spans. Each span is represented as a tuple :code:`(start, end)` corresponding
                to token indices.
        """
        query_spans = np.asarray(query_spans)
        chr_spans = np.asarray(chr_spans)
        # the query spans and the character spans
        overlap = compute_spans_overlap_matrix(
            source_spans=query_spans, target_spans=chr_spans, special_tokens=special_tokens_mask
        )
        # get begins and ends from mask
        tok_spans_begin = overlap.argmax(axis=1)
        tok_spans_end = tok_spans_begin + overlap.sum(axis=1)

        # exclude partially overlapping tokens at the beginning
        if not self.config.include_partial_start:
            partial_mask = chr_spans[tok_spans_begin, 0] != query_spans[:, 0]
            tok_spans_begin[partial_mask] += 1

        # exclude partially overlapping tokens at the end
        if not self.config.include_partial_end:
            partial_mask = chr_spans[tok_spans_end - 1, 1] != query_spans[:, 1]
            tok_spans_end[partial_mask] -= 1

        # build output
        return list(zip(tok_spans_begin, tok_spans_end, strict=True))
