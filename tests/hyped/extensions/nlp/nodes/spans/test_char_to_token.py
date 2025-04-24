from typing import Annotated

from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import Int, Len, Sequence
from hyped.extensions.nlp import CharToTokenSpans
from hyped.extensions.nlp.nodes.spans.utils import Spans


class TestCharToTokenSpans(BaseDataProcessorTest):
    processor = CharToTokenSpans()
    input_features = {
        "char_spans": Spans,
        "query_spans": Annotated[Spans, Len(3)],
    }
    input_data = [
        {
            "char_spans": [[0, 5], [6, 8], [9, 12], [13, 20]],
            "query_spans": [[0, 8], [9, 11], [7, 17]],
        }
    ]
    expected_output_feature = Annotated[Spans, Len(3, strict=True)]
    expected_output_data = [[[0, 2], [2, 3], [1, 4]]]


class TestCharToTokenSpansWithSpecialTokensMask(BaseDataProcessorTest):
    processor = CharToTokenSpans()
    input_features = {
        "char_spans": Spans,
        "query_spans": Annotated[Spans, Len(3)],
        "special_tokens_mask": Sequence[Int],
    }
    input_data = [
        {
            "char_spans": [[0, 5], [6, 8], [9, 12], [13, 20]],
            "query_spans": [[0, 8], [9, 11], [7, 17]],
            "special_tokens_mask": [1, 0, 0, 0],
        }
    ]
    expected_output_feature = Annotated[Spans, Len(3, strict=True)]
    expected_output_data = [[[1, 2], [2, 3], [1, 4]]]


class TestCharToTokenSpansNotIncludePartialStartEnd(BaseDataProcessorTest):
    processor = CharToTokenSpans(
        include_partial_start=False,
        include_partial_end=False,
    )
    input_features = {
        "char_spans": Spans,
        "query_spans": Annotated[Spans, Len(3)],
    }
    input_data = [
        {
            "char_spans": [[0, 5], [6, 8], [9, 12], [13, 20]],
            "query_spans": [[3, 8], [9, 12], [7, 17]],
        }
    ]
    expected_output_feature = Annotated[Spans, Len(3, strict=True)]
    expected_output_data = [[[1, 2], [2, 3], [2, 3]]]
