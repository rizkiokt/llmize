def initialize_llm(llm_model, api_key):
    if llm_model.startswith(("google/", "gemini", "gemma")):
        client = initialize_gemini(api_key)
    elif llm_model.startswith("openrouter/"):
        client = initialize_openrouter(api_key)
    else:
        client = initialize_huggingface(api_key)

    return client

def initialize_gemini(api_key):
    from google import genai

    client = genai.Client(api_key=api_key)

    return client

def initialize_huggingface(api_key):
    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=api_key)

    return client

def initialize_openrouter(api_key):
    """Initialize OpenRouter client with custom API format."""
    # OpenRouter uses direct requests, not a client library
    # We'll store the API key for use in generate_content_openrouter
    return {"api_key": api_key, "type": "openrouter"}

def initialize_openai(api_key):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    return client