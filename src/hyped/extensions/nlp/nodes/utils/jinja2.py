"""This module implements a Jinja2-based data processor for rendering templates.

It contains the implementation of a data processor that leverages the Jinja2
template engine to render input data into formatted output strings. The processor
takes a Jinja2 template, substitutes the provided values into the template, and
returns the rendered result. It allows flexible configuration of templates and
handles the substitution of dynamic input data into the templates during processing.
"""

from jinja2 import StrictUndefined, Template

from hyped.core import BaseDataProcessor, BaseDataProcessorConfig, RunContext, process_mode
from hyped.typing import Feature, String


class Jinja2ProcessorConfig(BaseDataProcessorConfig):
    """Configuration class for the :class:`Jinja2Processor`."""

    template: str
    """The Jinja2 template string to be used for rendering."""


class Jinja2Processor(BaseDataProcessor[Jinja2ProcessorConfig]):
    r"""A processor that renders data using a Jinja2 template.

    This processor takes input data, substitutes it into a Jinja2 template,
    and returns the rendered output as a string.

    **Example**:

    .. code-block:: python

        template = \"\"\"
        Hello, {{ name }}!
        Your score is {{ score }}.
        \"\"\"

    """

    def initialize(self, ctx: RunContext) -> None:
        """Initializes the processor by compiling the Jinja2 template.

        This method is called once when the processor is initialized.
        It sets up the template object for rendering.

        Args:
            ctx (RunContext): The runtime context for the processor.
        """
        self.template = Template(source=self.config.template, undefined=StrictUndefined)

    @process_mode(batched=False, backend="python")
    def process(self, ctx: RunContext, **kwargs: Feature) -> String:
        """Processes the input data and renders it using the Jinja2 template.

        This method is called for each input instance, substitutes the input
        data into the configured Jinja2 template, and returns the rendered result.

        Args:
            ctx (RunContext): The runtime context for the processor.
            **kwargs (Feature): Keyword arguments representing the input features
                to be rendered in the template.

        Returns:
            String: The rendered output as a string.
        """
        return self.template.render(**kwargs)
