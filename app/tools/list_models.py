"""Print the OpenAI models your API key can use:  python -m app.tools.list_models"""
from app.config import get_settings
from app.core.generation_client import GenerationClient, GenerationProviderError


def main() -> None:
    settings = get_settings()
    client = GenerationClient(settings)
    try:
        models = client.list_models()
    except GenerationProviderError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None
    configured = settings.openai_model
    print(f"Models available to your key ({len(models)}):")
    for m in models:
        mark = "  <-- configured" if m["name"] == configured else ""
        print(f"  {m['name']}  ({m['display_name']}){mark}")
    if configured and configured not in {m["name"] for m in models}:
        print(f"\nWARNING: configured OPENAI_MODEL '{configured}' is NOT in this list.")


if __name__ == "__main__":
    main()
