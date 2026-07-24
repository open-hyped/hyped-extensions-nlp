"""Hyped Natural Language Processing Extension.

Natural Language Processing Extension
"""

from typing import TYPE_CHECKING

from .__version__ import __version__, __version_tuple__

# list all imports
__all__ = [
    "CharToTokenSpans",
    "TransformersTokenizer",
    "SpansToMask",
    "Jinja2Processor",
    "ExtractSpans",
]

if TYPE_CHECKING:  # pragma: not covered
    # standard imports for static type checkers, linting and auto-completion
    # add your normal imports here
    from .nodes.spans.char_to_token import CharToTokenSpans
    from .nodes.spans.extract_spans import ExtractSpans
    from .nodes.spans.spans_to_mask import SpansToMask
    from .nodes.tokenizers.transformers import TransformersTokenizer
    from .nodes.utils.jinja2 import Jinja2Processor

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    # lazy imports
    _lazy_imports = {
        "CharToTokenSpans": "hyped.extensions.nlp.nodes.spans.char_to_token",
        "SpansToMask": "hyped.extensions.nlp.nodes.spans.spans_to_mask",
        "ExtractSpans": "hyped.extensions.nlp.nodes.spans.extract_spans",
        "TransformersTokenizer": "hyped.extensions.nlp.nodes.tokenizers.transformers",
        "Jinja2Processor": "hyped.extensions.nlp.nodes.utils.jinja2",
    }

    sys.modules[__name__] = LazyModule(
        __name__,
        __doc__,
        globals()["__file__"],
        __spec__,
        lazy_imports=_lazy_imports,
    )
    sys.modules[__name__].__version__ = __version__
    sys.modules[__name__].__version_tuple__ = __version_tuple__
