from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import String
from hyped.extensions.nlp import TransformersTokenizer
from hyped.extensions.nlp.nodes.tokenizers.transformers import TokenizerOutput


class TestTransformersTokenizer(BaseDataProcessorTest):
    processor = TransformersTokenizer(
        config=TransformersTokenizer.Config(tokenizer="bert-base-uncased")
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
        {"input_ids": [101, 7592, 2222, 2213, 1010, 2054, 1005, 1055, 2039, 1029, 102]}
    ]
