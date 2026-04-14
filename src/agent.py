from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

SYSTEM_PROMPT="""
You are a helpful assistant for a course. 

Use the search tool to find relevant information from the documentation before answering questions.

If you can find specific information through search, use it to provide accurate answers.
If the search doesn't return relevant results, let the user know and provide general guidance.
"""

def _build_model(model_name: str | None = None):
    name = model_name or "qwen2.5:0.5b"
    return OpenAIChatModel(model_name=name, provider=OllamaProvider(base_url='http://localhost:11434/v1'))


def create_agent(tools: list, model_name: str | None = None) -> Agent:
    """Create an Agent. If `model_name` is provided, use it instead of the default."""
    model = _build_model(model_name=model_name)
    return Agent(
        name="FAQ",
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        model=model
    )