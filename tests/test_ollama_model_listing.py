from app.backend.services.ollama_service import OllamaService


def test_available_downloaded_models_include_unknown_generators_but_exclude_embeddings():
    service = OllamaService()

    models = service._format_models_for_api([
        "qwen3:4b",
        "qwen3.5:0.8b",
        "qwen3.5:0.8b",
        "nomic-embed-text:latest",
        "all-minilm:latest",
    ])

    by_name = {model["model_name"]: model for model in models}
    assert by_name["qwen3.5:0.8b"] == {
        "display_name": "qwen3.5:0.8b",
        "model_name": "qwen3.5:0.8b",
        "provider": "Ollama",
    }
    assert by_name["qwen3:4b"]["display_name"] != "qwen3:4b"
    assert "nomic-embed-text:latest" not in by_name
    assert "all-minilm:latest" not in by_name
    assert len(models) == 2
