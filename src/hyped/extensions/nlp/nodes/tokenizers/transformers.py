"""This module implements the HuggingFace Transformers Tokenizer Processor.

It contains the implementation of a data processor for tokenizing text using
Transformers. The processor utilizes the Transformers library to tokenize
input text and produce various output features, such as input IDs, token types,
attention masks, special tokens masks, offset mappings, and word IDs. It offers
flexible configuration options to customize tokenization behavior and control
the output features generated during tokenization.
"""
from transformers import AutoTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy
from typing import Annotated
from hyped.core import process_mode, RunContext, BaseDataProcessor, BaseDataProcessorConfig
from hyped.core.features.features import SequenceFeature
from hyped.typing import Sequence, String, Mapping, Int32, Len, ExcludeFieldIf, FeatureResolver, FeatureValidator
import pyarrow as pa


class TransformersTokenizerConfig(BaseDataProcessorConfig):
    """Configuration for the Transformers Tokenizer processor."""

    tokenizer: str
    """The name or path of the pre-trained tokenizer."""

    add_special_tokens: bool = True
    """Whether to add special tokens during tokenization."""

    padding: bool | str | PaddingStrategy = False
    """Padding strategy for sequences."""

    truncation: bool | str | TruncationStrategy = False
    """Truncation strategy for sequences."""

    max_length: None | int = None
    """Maximum sequence length after tokenization."""

    stride: int = 0
    """Stride for tokenization."""

    is_split_into_words: bool = False
    """Flag indicating whether inputs are already split into words."""

    pad_to_multiple_of: None | int = None
    """Pad tokenized sequences to a multiple of this value."""

    return_tokens: bool = False
    """Whether to include tokenized tokens in the output."""

    return_token_type_ids: bool = False
    """Whether to include token type IDs in the output."""

    return_attention_mask: bool = False
    """Whether to include attention masks in the output."""

    return_special_tokens_mask: bool = False
    """Whether to include special tokens masks in the output."""

    return_offsets_mapping: bool = False
    """Whether to include offsets mappings in the output."""

    return_length: bool = False
    """Whether to include the length of sequences in the output."""

    return_word_ids: bool = False
    """Whether to include word IDs in the output."""


def _get_output_sequence_length(config: TransformersTokenizerConfig) -> int | None:
    """Determine the sequence length based on tokenizer configuration."""
    # check for constant length
    is_constant = (
        (config.max_length is not None)
        and (config.padding == "max_length")
        and (config.truncation in (True, "longest_first", "only_first", "only_second"))
    )
    # get sequence length in case it's constant
    return config.max_length if is_constant else None


class TokenizerOutput(Mapping):
    input_ids: Annotated[Sequence[Int32], FeatureResolver(lambda c, i, _: Annotated[Sequence[Int32], Len(_get_output_sequence_length(c))])]
    tokens: Annotated[Sequence[String] | None, ExcludeFieldIf(lambda c, i, _: not c.return_tokens), FeatureResolver(lambda c, i, _: Annotated[Sequence[String], Len(_get_output_sequence_length(c))])]
    token_type_ids: Annotated[Sequence[Int32] | None, ExcludeFieldIf(lambda c, i, _: not c.return_token_type_ids), FeatureResolver(lambda c, i, _: Annotated[Sequence[Int32], Len(_get_output_sequence_length(c))])]
    attention_mask: Annotated[Sequence[Int32] | None, ExcludeFieldIf(lambda c, i, _: not c.return_attention_mask), FeatureResolver(lambda c, i, _: Annotated[Sequence[Int32], Len(_get_output_sequence_length(c))])]
    special_tokens_mask: Annotated[Sequence[Int32] | None, ExcludeFieldIf(lambda c, i, _: not c.return_special_tokens_mask), FeatureResolver(lambda c, i, _: Annotated[Sequence[Int32], Len(_get_output_sequence_length(c))])]
    offset_mapping: Annotated[Sequence[Annotated[Sequence[Int32], Len(2)]] | None, ExcludeFieldIf(lambda c, i, _: not c.return_offsets_mapping), FeatureResolver(lambda c, i, _: Annotated[Sequence[Annotated[Sequence[Int32], Len(2)]], Len(_get_output_sequence_length(c))])]
    length: Annotated[Int32 | None, ExcludeFieldIf(lambda c, i, _: not c.return_length)]
    word_ids: Annotated[Sequence[Int32] | None, ExcludeFieldIf(lambda c, i, _: not c.return_word_ids), FeatureResolver(lambda c, i, _: Annotated[Sequence[Int32], Len(_get_output_sequence_length(c))])]


def _validate_text_type(feature: String | Sequence[String], config: TransformersTokenizerConfig, session):
    if config.is_split_into_words:
        if isinstance(feature, String):
            raise TypeError(
                f"Expects a list of pre-tokenized words "
                "when `is_split_into_words=True`. You possibly "
                "passed the input text as a single string."
            )
    elif isinstance(feature, SequenceFeature):
        raise TypeError(
                f"Expects a string as input "
                "when `is_split_into_words=False`. Got a Sequence of strings."
            )
    return feature


class TransformersTokenizer(BaseDataProcessor[TransformersTokenizerConfig]):
    """Transformers Tokenizer data processor.

    This processor tokenizes input text using a specified tokenizer.
    """

    def __init__(self, config: None | TransformersTokenizerConfig = None, **kwargs) -> None:
        """Initialize the Transformers Tokenizer processor.

        Args:
            config (TransformersTokenizerConfig): Processor configuration.
            **kwargs: Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        super(TransformersTokenizer, self).__init__(config, **kwargs)
        # load the tokenizer instance
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer, use_fast=True, add_prefix_space=True
        )

    @process_mode(batched=True, backend="arrow")
    def process(
        self,
        ctx: RunContext,
        text: Annotated[String | Sequence[String], FeatureValidator(_validate_text_type)],
        text_pair: Annotated[String | Sequence[String] | None, FeatureValidator(_validate_text_type)] = None,
        text_target: Annotated[String | Sequence[String] | None, FeatureValidator(_validate_text_type)] = None,
        text_pair_target: Annotated[String | Sequence[String] | None, FeatureValidator(_validate_text_type)] = None,
    ) -> TokenizerOutput:
        # apply tokenizer
        enc = self.tokenizer(
            text=text.to_pylist(),
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=self.config.add_special_tokens,
            padding=self.config.padding,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            stride=self.config.stride,
            is_split_into_words=self.config.is_split_into_words,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            return_token_type_ids=self.config.return_token_type_ids,
            return_attention_mask=self.config.return_attention_mask,
            return_special_tokens_mask=self.config.return_special_tokens_mask,
            return_offsets_mapping=self.config.return_offsets_mapping,
            return_length=self.config.return_length,
        )
        # convert tokenizer BatchEncoding to dict (of lists)
        out = dict(enc)
        # include additional features
        if self.config.return_tokens:
            out["tokens"] = list(map(self.tokenizer.convert_ids_to_tokens, enc.input_ids))
        if self.config.return_word_ids:
            out["word_ids"] = [
                [(i if i is not None else -1) for i in enc.word_ids(j)] for j in range(len(ctx.index))
            ]
        # convert dict-of-lists to arrow StructArray
        return pa.table(out, schema=ctx.output_type.arrow_schema).to_struct_array()
