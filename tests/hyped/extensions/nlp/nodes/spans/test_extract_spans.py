from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import Int, Bool, Sequence
from hyped.extensions.nlp import ExtractSpans
from hyped.extensions.nlp.nodes.spans.utils import Spans

class TestExtractSpans(BaseDataProcessorTest):
    processor = ExtractSpans()
    input_features = {
        "sequence": Sequence[Int],
        "begin_marker": Int,
        "end_marker": Int
    }
    input_data = [
        {
            "sequence": [0, 0, 1, 0, 2, 0, 0],
            "begin_marker": 1,
            "end_marker": 2
        }
    ]
    expected_output_feature = Spans
    expected_output_data = [[[2, 5]]]

class TestExtractSpansWithOverlap(BaseDataProcessorTest):
    processor = ExtractSpans()
    input_features = {
        "sequence": Sequence[Int],
        "begin_marker": Int,
        "end_marker": Int
    }
    input_data = [
        {
            "sequence": [0, 1, 0, 1, 2, 0, 2],
            "begin_marker": 1,
            "end_marker": 2
        }
    ]
    expected_output_feature = Spans
    expected_output_data = [[[1, 5], [3, 7]]]

class TestExtractSpansNoHit(BaseDataProcessorTest):
    processor = ExtractSpans()
    input_features = {
        "sequence": Sequence[Int],
        "begin_marker": Int,
        "end_marker": Int
    }
    input_data = [
        {
            "sequence": [0, 0, 0, 0, 0, 0, 0],
            "begin_marker": 1,
            "end_marker": 2
        }
    ]
    expected_output_feature = Spans
    expected_output_data = [[]]

class TestExtractSpansErrorOnMismatch(BaseDataProcessorTest):
    processor = ExtractSpans()
    input_features = {
        "sequence": Sequence[Int],
        "begin_marker": Int,
        "end_marker": Int
    }
    input_data = [
        {
            "sequence": [0, 0, 1, 2, 1, 0, 0],
            "begin_marker": 1,
            "end_marker": 2
        }
    ]
    expected_execution_error = ValueError

class TestExtractSpansErrorOnInvalid(BaseDataProcessorTest):
    processor = ExtractSpans()
    input_features = {
        "sequence": Sequence[Int],
        "begin_marker": Int,
        "end_marker": Int
    }
    input_data = [
        {
            "sequence": [0, 0, 0, 2, 1, 0, 0],
            "begin_marker": 1,
            "end_marker": 2
        }
    ]
    expected_execution_error = ValueError

class TestExtractSpansUnclosedFinal(BaseDataProcessorTest):
    processor = ExtractSpans(allow_unclosed_final_span=True)
    input_features = {
        "sequence": Sequence[Int],
        "begin_marker": Int,
        "end_marker": Int,
    }
    input_data = [
        {
            "sequence": [0, 0, 1, 0, 0, 0],
            "begin_marker": 1,
            "end_marker": 2,
        }
    ]
    expected_output_feature = Spans
    expected_output_data = [[[2, 6]]]


class TestExtractSpansUnclosedInitial(BaseDataProcessorTest):
    processor = ExtractSpans(allow_unclosed_initial_span=True)
    input_features = {
        "sequence": Sequence[Int],
        "begin_marker": Int,
        "end_marker": Int,
    }
    input_data = [
        {
            "sequence": [0, 0, 2, 0, 0, 0],
            "begin_marker": 1,
            "end_marker": 2,
        }
    ]
    expected_output_feature = Spans
    expected_output_data = [[[0, 3]]]


class TestExtractSpansUnclosed(BaseDataProcessorTest):
    processor = ExtractSpans(
        allow_unclosed_initial_span=True,
        allow_unclosed_final_span=True
    )
    input_features = {
        "sequence": Sequence[Int],
        "begin_marker": Int,
        "end_marker": Int,
    }
    input_data = [
        {
            "sequence": [0, 0, 2, 0, 1, 0],
            "begin_marker": 1,
            "end_marker": 2,
        }
    ]
    expected_output_feature = Spans
    expected_output_data = [[[0, 3], [4, 6]]]
