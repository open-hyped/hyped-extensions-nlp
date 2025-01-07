from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import Sequence, String
from hyped.extensions.nlp import TransformersTokenizer
from hyped.extensions.nlp.nodes.tokenizers.transformers import TokenizerOutput


class TestTransformersTokenizer(BaseDataProcessorTest):
    processor = TransformersTokenizer(tokenizer="./tests/artifacts/tokenizers/bert-base-uncased")
    input_features = {
        "text": String,
    }
    input_data = [
        {
            "text": "Hello LLM, what's up?",
        }
    ]
    expected_output_feature = TokenizerOutput
    expected_output_data = [
        {"input_ids": [101, 7592, 2222, 2213, 1010, 2054, 1005, 1055, 2039, 1029, 102]}
    ]


class TestTransformersTokenizerAllFeatures(BaseDataProcessorTest):
    processor = TransformersTokenizer(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased",
        return_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
        return_length=True,
        return_word_ids=True,
    )
    input_features = {
        "text": String,
    }
    input_data = [
        {
            "text": "Hello LLM, what's up?",
        }
    ]
    expected_output_feature = TokenizerOutput
    expected_output_data = [
        {
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "input_ids": [101, 7592, 2222, 2213, 1010, 2054, 1005, 1055, 2039, 1029, 102],
            "length": 11,
            "offset_mapping": [
                [0, 0],
                [0, 5],
                [6, 8],
                [8, 9],
                [9, 10],
                [11, 15],
                [15, 16],
                [16, 17],
                [18, 20],
                [20, 21],
                [0, 0],
            ],
            "special_tokens_mask": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "tokens": ["[CLS]", "hello", "ll", "##m", ",", "what", "'", "s", "up", "?", "[SEP]"],
            "word_ids": [-1, 0, 1, 1, 2, 3, 4, 5, 6, 7, -1],
        }
    ]


class TestTransformersTokenizerSplitIntoWords(BaseDataProcessorTest):
    processor = TransformersTokenizer(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased", is_split_into_words=True
    )
    input_features = {
        "text": Sequence[String],
    }
    input_data = [{"text": ["Hello", "LLM", ",", "what's", "up", "?"]}]
    expected_output_feature = TokenizerOutput
    expected_output_data = [
        {"input_ids": [101, 7592, 2222, 2213, 1010, 2054, 1005, 1055, 2039, 1029, 102]}
    ]


class TestTransformersTokenizerSplitIntoWordsError(BaseDataProcessorTest):
    processor = TransformersTokenizer(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased", is_split_into_words=True
    )
    input_features = {
        "text": String,
    }
    input_data = [
        {
            "text": "Hello LLM, what's up?",
        }
    ]
    expected_verification_error = TypeError


class TestTransformersTokenizerMaxLengthPadding(BaseDataProcessorTest):
    processor = TransformersTokenizer(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased",
        padding="max_length",
        truncation=True,
        max_length=15,
    )
    input_features = {
        "text": String,
    }
    input_data = [
        {
            "text": "Hello LLM, what's up?",
        }
    ]
    expected_output_feature = TokenizerOutput
    expected_output_data = [
        {"input_ids": [101, 7592, 2222, 2213, 1010, 2054, 1005, 1055, 2039, 1029, 102, 0, 0, 0, 0]}
    ]

    def execute_test(self):
        output_feature: TokenizerOutput = super().execute_test()[0]
        assert output_feature["input_ids"].dtype.length == 15
