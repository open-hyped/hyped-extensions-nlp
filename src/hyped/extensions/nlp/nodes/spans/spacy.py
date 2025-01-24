from typing import Any

import spacy
from typing_extensions import Annotated

from hyped.core import BaseDataProcessor, BaseDataProcessorConfig, RunContext
from hyped.typing import ExcludeFieldIf, Int, Mapping, Sequence, String


class SpacyProcessorConfig(BaseDataProcessorConfig):
    model: str
    return_tokens: bool = True
    return_sentences: bool = False
    return_entities: bool = False


class Token(Mapping):
    begin: Int
    end: Int
    pos: String
    lemma: String


class Sentence(Mapping):
    begin: Int
    end: Int


class Entity(Mapping):
    begin: Int
    end: Int
    entityType: String


class SpacyProcessorOutput(Mapping):
    tokens: Annotated[Sequence[Token], ExcludeFieldIf(lambda c, i, s: not c.return_tokens)]
    sentences: Annotated[Sequence[Sentence], ExcludeFieldIf(lambda c, i, s: not c.return_sentences)]
    entities: Annotated[Sequence[Entity], ExcludeFieldIf(lambda c, i, s: not c.return_entities)]


class SpacyProcessor(BaseDataProcessor[SpacyProcessorConfig]):
    def initialize(self, ctx):
        self.nlp = spacy.load(self.config.model)

    def process(
        self,
        ctx: RunContext,
        text: String,
    ) -> SpacyProcessorOutput:
        doc = self.nlp(text)

        result = {}
        if self.config.return_tokens:
            result["tokens"] = self.get_tokens(doc)
        if self.config.return_sentences:
            result["sentences"] = self.get_sentences(doc)
        if self.config.return_entities:
            result["entities"] = self.get_entities(doc)

        return result

    def get_tokens(self, doc: Any) -> list[dict[str, Any]]:
        tokens = []
        for token in doc:
            tokens.append(
                dict(
                    begin=token.idx,
                    end=token.idx + len(token.text),
                    pos=token.tag_,
                    lemma=token.lemma_,
                )
            )
        return tokens

    def get_sentences(self, doc: Any) -> list[dict[str, Any]]:
        sents = []
        for sent in doc.sents:
            sents.append(
                dict(
                    begin=sent.start_char,
                    end=sent.end_char,
                    source="preprocessor",
                )
            )
        return sents

    def get_entities(self, doc: Any) -> list[dict[str, Any]]:
        ents = []
        for ent in doc.ents:
            ents.append(
                dict(
                    begin=ent.start_char,
                    end=ent.end_char,
                    entityType=ent.label_,
                    source="preprocessor",
                )
            )
        return ents
