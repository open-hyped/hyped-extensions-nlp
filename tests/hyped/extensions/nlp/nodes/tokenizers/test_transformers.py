from unittest.mock import MagicMock, patch

from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import Sequence, String
from hyped.extensions.nlp import TransformersTokenizer
from hyped.extensions.nlp.nodes.tokenizers.transformers import (
    ApplyChatTemplate,
    Message,
    TokenizerOutput,
)


class TestTransformersTokenizer(BaseDataProcessorTest):
    processor = TransformersTokenizer(tokenizer="./tests/artifacts/tokenizers/bert-base-uncased")
    input_features = {"text": String}
    input_data = [
        {"text": "Hello LLM, what's up?"},
    ]
    expected_output_feature = TokenizerOutput
    expected_output_data = [
        {"input_ids": [101, 7592, 2222, 2213, 1010, 2054, 1005, 1055, 2039, 1029, 102]},
    ]

    @patch("hyped.extensions.nlp.nodes.tokenizers.transformers.ApplyChatTemplate")
    def test_apply_chat_template(self, mock_apply_chat_template_type: MagicMock) -> None:
        processor = type(self).processor
        # create mock inputs
        mock_conversation = MagicMock()
        mock_add_generation_prompt = MagicMock()
        mock_continue_final_message = MagicMock()
        # call the function
        out = processor.apply_chat_template(
            conversation=mock_conversation,
            add_generation_prompt=mock_add_generation_prompt,
            continue_final_message=mock_continue_final_message,
        )
        # check call
        assert out == mock_apply_chat_template_type.return_value.call.return_value
        mock_apply_chat_template_type.assert_called_once_with(
            tokenizer=processor.config.tokenizer,
            add_generation_prompt=mock_add_generation_prompt,
            continue_final_message=mock_continue_final_message,
        )


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
    input_features = {"text": String}
    input_data = [
        {"text": "Hello LLM, what's up?"},
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
        },
    ]


class TestTransformersTokenizerTextPairTarget(BaseDataProcessorTest):
    processor = TransformersTokenizer(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased",
        max_length=20,
        padding="max_length",
        truncation=True,
    )
    input_features = {
        "text": String,
        "text_pair": String,
        "text_target": String,
        "text_pair_target": String,
    }
    input_data = [
        {
            "text": "This is sentence A.",
            "text_pair": "This is sentence B.",
            "text_target": "Das ist Satz A.",
            "text_pair_target": "Das ist Satz B.",
        }
    ]
    expected_output_feature = TokenizerOutput
    expected_output_data = [
        {
            "input_ids": [
                101,
                2023,
                2003,
                6251,
                1037,
                1012,
                102,
                2023,
                2003,
                6251,
                1038,
                1012,
                102,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "labels": [
                101,
                8695,
                21541,
                2938,
                2480,
                1037,
                1012,
                102,
                8695,
                21541,
                2938,
                2480,
                1038,
                1012,
                102,
                0,
                0,
                0,
                0,
                0,
            ],
        }
    ]


class TestTransformersTokenizerSplitIntoWords(BaseDataProcessorTest):
    processor = TransformersTokenizer(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased", is_split_into_words=True
    )
    input_features = {"text": Sequence[String]}
    input_data = [
        {"text": ["Hello", "LLM", ",", "what's", "up", "?"]},
    ]
    expected_output_feature = TokenizerOutput
    expected_output_data = [
        {"input_ids": [101, 7592, 2222, 2213, 1010, 2054, 1005, 1055, 2039, 1029, 102]},
    ]


class TestTransformersTokenizerSplitIntoWordsError(BaseDataProcessorTest):
    processor = TransformersTokenizer(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased", is_split_into_words=True
    )
    input_features = {"text": String}
    input_data = [
        {"text": "Hello LLM, what's up?"},
    ]
    expected_verification_error = TypeError


class TestTransformersTokenizerMaxLengthPadding(BaseDataProcessorTest):
    processor = TransformersTokenizer(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased",
        padding="max_length",
        truncation=True,
        max_length=15,
    )
    input_features = {"text": String}
    input_data = [
        {"text": "Hello LLM, what's up?"},
    ]
    expected_output_feature = TokenizerOutput
    expected_output_data = [
        {"input_ids": [101, 7592, 2222, 2213, 1010, 2054, 1005, 1055, 2039, 1029, 102, 0, 0, 0, 0]},
    ]

    def execute_test(self):
        output_feature: TokenizerOutput = super().execute_test()[0]
        assert output_feature["input_ids"].dtype.length == 15


class TestTransformersTokenizerApplyChatTemplate(BaseDataProcessorTest):
    processor = ApplyChatTemplate(tokenizer="./tests/artifacts/tokenizers/mistral-7b-instruct-v0.3")
    input_features = {"conversation": Sequence[Message]}
    input_data = [
        {
            "conversation": [
                {"role": "user", "content": "Hello from User"},
                {"role": "assistant", "content": "Hello from Assistant"},
            ]
        }
    ]
    expected_output_feature = String
    expected_output_data = ["<s>[INST] Hello from User[/INST] Hello from Assistant</s>"]


class TestTransformersTokenizerApplyChatTemplateAddGenerationPrompt(BaseDataProcessorTest):
    processor = ApplyChatTemplate(
        tokenizer="./tests/artifacts/tokenizers/mistral-7b-instruct-v0.3",
        add_generation_prompt=True,
    )
    input_features = {"conversation": Sequence[Message]}
    input_data = [
        {
            "conversation": [
                {"role": "user", "content": "Hello from User"},
            ]
        }
    ]
    expected_output_feature = String
    expected_output_data = ["<s>[INST] Hello from User[/INST]"]


class TestTransformersTokenizerApplyChatTemplateContinueFinalMessage(BaseDataProcessorTest):
    processor = ApplyChatTemplate(
        tokenizer="./tests/artifacts/tokenizers/mistral-7b-instruct-v0.3",
        continue_final_message=True,
    )
    input_features = {"conversation": Sequence[Message]}
    input_data = [
        {
            "conversation": [
                {"role": "user", "content": "Hello from User"},
                {"role": "assistant", "content": "Hello from Assistant"},
            ]
        }
    ]
    expected_output_feature = String
    expected_output_data = ["<s>[INST] Hello from User[/INST] Hello from Assistant"]
