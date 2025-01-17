from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.extensions.nlp import Jinja2Processor
from hyped.typing import Int, String


class TestJinja2Processor(BaseDataProcessorTest):
    processor = Jinja2Processor(template="Value: {{ value }}")
    input_features = {"value": Int}
    input_data = [{"value": 0}, {"value": 1}, {"value": 2}, {"value": 3}]
    expected_output_feature = String
    expected_output_data = ["Value: 0", "Value: 1", "Value: 2", "Value: 3"]


class TestJinja2ProcessorWithConditionalLogic(BaseDataProcessorTest):
    processor = Jinja2Processor(template="{% if value > 2 %}Large{% else %}Small{% endif %}")
    input_features = {"value": Int}
    input_data = [{"value": 0}, {"value": 1}, {"value": 2}, {"value": 3}]
    expected_output_feature = String
    expected_output_data = ["Small", "Small", "Small", "Large"]
