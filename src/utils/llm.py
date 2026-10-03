"""Helper functions for LLM"""

import json
import os
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info


def _lookup_ollama_model_digest(model_name: str) -> str | None:
    """Best-effort local /api/tags lookup; returns no digest on any failure."""
    try:
        from ollama import Client

        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
        response = Client(host=base_url, timeout=1.0).list()
        for model in getattr(response, "models", []):
            installed_name = getattr(model, "model", None) or getattr(model, "name", None)
            if installed_name == model_name:
                return getattr(model, "digest", None)
    except Exception:
        return None
    return None
from src.utils.progress import progress
from src.graph.state import AgentState
from src.run_context import RunStopped, get_active_run_context


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 3,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    
    # Extract model configuration if state is provided and agent_name is available
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        # Use system defaults when no state or agent_name is provided
        model_name = "gpt-4.1"
        model_provider = "OPENAI"

    # Extract API keys from state if available
    api_keys = None
    if state:
        request = state.get("metadata", {}).get("request")
        if request and hasattr(request, 'api_keys'):
            api_keys = request.api_keys

    metadata = state.get("metadata", {}) if state else {}
    strict_execution = bool(metadata.get("strict_execution"))
    run_context = metadata.get("run_context") or (get_active_run_context() if strict_execution else None)
    local_ollama = str(getattr(model_provider, "value", model_provider)).lower() == "ollama"
    model_timeout = min(30.0, run_context.remaining_seconds()) if run_context is not None else None
    run_options = {}
    if strict_execution and local_ollama:
        run_options = {
            "temperature": 0,
            "reasoning": False,
            "num_predict": 768,
            "timeout_seconds": model_timeout,
            "structured_output": "json_schema",
        }
    if run_context is not None:
        run_context.check_active()
        model_digest = (
            run_context.get_model_digest(model_name, _lookup_ollama_model_digest)
            if strict_execution and local_ollama
            else None
        )
        run_context.capture_model(
            model_name,
            model_provider,
            prompt,
            schema=pydantic_model.model_json_schema(),
            temperature=0 if strict_execution and local_ollama else None,
            options=run_options,
            model_digest=model_digest,
        )
    model_info = get_model_info(model_name, model_provider)
    llm = get_model(
        model_name,
        model_provider,
        api_keys,
        timeout_seconds=model_timeout,
        strict_run=strict_execution,
    )

    # For non-JSON support models, we can use structured output
    if strict_execution and local_ollama:
        llm = llm.with_structured_output(pydantic_model, method="json_schema")
    elif not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            if run_context is not None:
                run_context.check_active()
            # Call the LLM
            result = llm.invoke(prompt)
            if result is None:
                raise ValueError("LLM returned an empty response")

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
                raise ValueError("LLM returned invalid or empty structured output")
            else:
                if strict_execution and not isinstance(result, pydantic_model):
                    if isinstance(result, dict):
                        return pydantic_model.model_validate(result)
                    raise ValueError("LLM returned an unexpected structured output type")
                return result

        except RunStopped:
            raise
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                if strict_execution:
                    raise RuntimeError(f"LLM analysis failed for {agent_name or 'agent'}: {e}") from e
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    request = state.get("metadata", {}).get("request")
    
    if request and hasattr(request, 'get_agent_model_config'):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        # Ensure we have valid values
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, 'value') else str(model_provider)
    
    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "gpt-4.1"
    model_provider = state.get("metadata", {}).get("model_provider") or "OPENAI"
    
    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider
