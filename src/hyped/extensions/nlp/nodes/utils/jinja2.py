from hyped.core import BaseDataProcessor, BaseDataProcessorConfig, RunContext, process_mode
from hyped.typing import Feature, String

from jinja2 import Template

class Jinja2ProcessorConfig(BaseDataProcessorConfig):
    """Configuration class for the :class:`Jinja2Processor`."""

    template: str
    """The Jinja2 template string to be used for rendering."""


class Jinja2Processor(BaseDataProcessor[Jinja2ProcessorConfig]):
    """A processor that renders data using a Jinja2 template.

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
            ctx (RunContext): The runtime context for the processor, providing environment information.
        """
        self.template = Template(source=self.config.template)

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