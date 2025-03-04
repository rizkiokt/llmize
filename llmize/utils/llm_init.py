from google import genai
from huggingface_hub import InferenceClient

def initialize_gemini(api_key):
    client = genai.Client(api_key)

    return client

def initialize_huggingface(api_key):
    client = InferenceClient(api_key)

    return client