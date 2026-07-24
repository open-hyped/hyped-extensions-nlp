from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import Bool, Int, Sequence
from hyped.extensions.nlp import SpansToMask
from hyped.extensions.nlp.nodes.spans.utils import Spans


class TestSpansToMask(BaseDataProcessorTest):
    processor = SpansToMask()
    input_features = {
        "spans": Spans,
        "length": Int,
    }
    input_data = [
        {
            "spans": [[0, 3], [6, 7]],
            "length": 10,
        }
    ]
    expected_output_feature = Sequence[Bool]
    expected_output_data = [[True, True, True, False, False, False, True, False, False, False]]


class TestSpansToMaskWithEmptySpans(BaseDataProcessorTest):
    processor = SpansToMask()
    input_features = {
        "spans": Spans,
        "length": Int,
    }
    input_data = [
        {
            "spans": [],
            "length": 10,
        },
        {
            "spans": [[3, 3]],
            "length": 10,
        },
    ]
    expected_output_feature = Sequence[Bool]
    expected_output_data = [
        [False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False],
    ]


class TestSpansToMaskWithOverlap(BaseDataProcessorTest):
    processor = SpansToMask()
    input_features = {
        "spans": Spans,
        "length": Int,
    }
    input_data = [
        {
            "spans": [[0, 3], [2, 7]],
            "length": 10,
        }
    ]
    expected_output_feature = Sequence[Bool]
    expected_output_data = [[True, True, True, True, True, True, True, False, False, False]]


class TestSpansToMaskFullSequence(BaseDataProcessorTest):
    processor = SpansToMask()
    input_features = {
        "spans": Spans,
        "length": Int,
    }
    input_data = [
        {
            "spans": [[0, 10]],
            "length": 10,
        }
    ]
    expected_output_feature = Sequence[Bool]
    expected_output_data = [[True, True, True, True, True, True, True, True, True, True]]
