def initialize_llm(llm_model, api_key):
    if llm_model.startswith(("gemini", "gemma")):
        client = initialize_gemini(api_key)
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

def initialize_openai(api_key):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    return client