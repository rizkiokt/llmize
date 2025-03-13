from google import genai
from huggingface_hub import InferenceClient
from openai import OpenAI


def initialize_llm(llm_model, api_key):
    if llm_model.startswith("gemini"):
        client = initialize_gemini(api_key)
    else:
        client = initialize_huggingface(api_key)

    return client

def initialize_gemini(api_key):
    client = genai.Client(api_key=api_key)

    return client

def initialize_huggingface(api_key):
    client = InferenceClient(api_key=api_key)

    return client

def initialize_openai(api_key):
    client = OpenAI(api_key=api_key)

    return client