import json
import os
from typing import List, Dict
from anthropic import Anthropic


# Setup environment variable for Anthropic API key
def setup_environment():
    """
    Set up the ANTHROPIC_API_KEY environment variable.
    Replace 'your-api-key-here' with your actual Anthropic API key.
    """
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"


# Initialize environment on import
setup_environment()


def get_anthropic_key():
    """
    Get the Anthropic API key from environment variables.
    """
    return os.environ.get("ANTHROPIC_API_KEY", "")


def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float = None,
    temperature: float = None,
    max_tokens: int = 500,
    model: str = "claude-haiku-4-5",
    anthropic_api_key=None,
    **kwargs,
):
    """
    Generate text using Anthropic's Claude API with a single input prompt.

    Args:
        prompt: The input text prompt to send to the model
        role: The role of the message (default: "user")
        top_p: Nucleus sampling parameter
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        model: Model identifier (default: claude-haiku-4-5)
        anthropic_api_key: Optional API key for Anthropic (uses env variable if not provided)
        **kwargs: Additional parameters to pass to the API

    Returns:
        Dictionary with 'role' and 'content' keys
    """
    if anthropic_api_key is None:
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    client = Anthropic(api_key=anthropic_api_key)

    # Anthropic API requires "user" role for messages we're sending
    # The role parameter is kept for compatibility but always use "user" for input
    effective_role = "user"

    # Build request parameters
    request_params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": effective_role, "content": prompt}],
    }

    # Add optional parameters if provided
    if temperature is not None:
        request_params["temperature"] = temperature
    if top_p is not None:
        request_params["top_p"] = top_p

    # Add any additional kwargs
    request_params.update(kwargs)

    try:
        response = client.messages.create(**request_params)
        output_dict = {
            "role": "assistant",
            "content": response.content[0].text,
        }
    except Exception as e:
        raise Exception(f"Failed to get correct output from LLM call. Error: {e}")

    return output_dict


def generate_with_multiple_input(
    messages: List[Dict],
    top_p: float = None,
    temperature: float = None,
    max_tokens: int = 500,
    model: str = "claude-haiku-4-5",
    anthropic_api_key=None,
    **kwargs,
):
    """
    Generate text using Anthropic's Claude API with multiple input messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        top_p: Nucleus sampling parameter
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        model: Model identifier (default: claude-haiku-4-5)
        anthropic_api_key: Optional API key for Anthropic (uses env variable if not provided)
        **kwargs: Additional parameters to pass to the API

    Returns:
        Dictionary with 'role' and 'content' keys
    """
    if anthropic_api_key is None:
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    client = Anthropic(api_key=anthropic_api_key)

    # Build request parameters
    request_params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    # Add optional parameters if provided
    if temperature is not None:
        request_params["temperature"] = temperature
    if top_p is not None:
        request_params["top_p"] = top_p

    # Add any additional kwargs
    request_params.update(kwargs)

    try:
        response = client.messages.create(**request_params)
        output_dict = {
            "role": "assistant",
            "content": response.content[0].text,
        }
    except Exception as e:
        raise Exception(
            f"Failed to get correct output dict. Please try again. Error: {e}"
        )

    return output_dict
