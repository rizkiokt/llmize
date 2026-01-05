import time
import requests
import json
from google import genai

from ..utils.logger import log_info, log_warning, log_error
from ..utils.decorators import time_it

#@time_it
def generate_content(client, model, prompt, temperature=None, max_retries=None, retry_delay=None):
    """
    Generate content using the specified language model.

    This function delegates the content generation to either the Gemini or Hugging Face model
    based on the provided model name.

    Parameters:
    - client: The API client used to interact with the language model.
    - model (str): The name of the language model to use.
    - prompt (str): The textual prompt to generate content from.
    - temperature (float, optional): Controls the creativity of the responses (default from config).
    - max_retries (int, optional): Maximum number of retry attempts in case of rate-limiting (default from config).
    - retry_delay (int, optional): Delay in seconds between retry attempts (default from config).

    Returns:
    - str: The generated content as a string, or None if the request was unsuccessful after retries.
    """
    # Import config here to avoid circular imports
    from ..config import get_config
    config = get_config()
    
    # Use config defaults if not provided
    if temperature is None:
        temperature = config.temperature
    if max_retries is None:
        max_retries = config.max_retries
    if retry_delay is None:
        retry_delay = config.retry_delay

    if model.startswith(("google/", "gemini", "gemma")):
        return generate_content_gemini(client, model, prompt, temperature, max_retries, retry_delay)
    elif model.startswith("openrouter/"):
        return generate_content_openrouter(client, model, prompt, temperature, max_retries, retry_delay)
    else:
        return generate_content_huggingface(client, model, prompt, temperature, max_retries, retry_delay)    

def generate_content_gemini(client, model, prompt, temperature, max_retries=10, retry_delay=5):
    # Strip google/ prefix if present for backward compatibility
    if model.startswith("google/"):
        model = model[6:]  # Remove "google/" prefix
    
    generate_content_config = genai.types.GenerateContentConfig(temperature=temperature)
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=generate_content_config)
            return response.text  # If the request is successful, return the response
        except Exception as e:
            if 'RESOURCE_EXHAUSTED' in str(e):  # You can check for rate-limiting specific message in the error
                log_warning(f"LLM rate limit hit, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                log_error(f"An error occurred: {e}")
                break  # If it's not a rate-limiting error, stop retrying
    log_warning("Max retries reached. Could not complete the request.")
    return None  # Return None if the max retries are reached and the request was unsuccessful

def generate_content_huggingface(client, model, prompt, temperature, max_retries=10, retry_delay=5):
    messages = [
        { "role": "user", "content": prompt}
    ]
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature)
            response = completion.choices[0].message.content
            return response  # If the request is successful, return the response
        except Exception as e:
            if 'RESOURCE_EXHAUSTED' in str(e):  # You can check for rate-limiting specific message in the error
                log_warning(f"LLM rate limit hit, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                log_error(f"An error occurred: {e}")
                break  # If it's not a rate-limiting error, stop retrying
    log_warning("Max retries reached. Could not complete the request.")
    return None  # Return None if the max retries are reached and the request was unsuccessful

def generate_content_openrouter(client, model, prompt, temperature, max_retries=10, retry_delay=5):
    """Generate content using OpenRouter API."""
    # Extract API key from client dict
    api_key = client["api_key"]
    
    # Prepare the request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional headers for rankings on openrouter.ai
        "HTTP-Referer": "https://github.com/rizkiokt/llmize",
        "X-Title": "LLMize Optimization Library",
    }
    data = {
        "model": model,  # Use the full model name with openrouter/ prefix stripped
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature
    }
    
    # Strip the openrouter/ prefix for the API call
    if model.startswith("openrouter/"):
        data["model"] = model[11:]  # Remove "openrouter/" prefix
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            if 'rate limit' in str(e).lower() or '429' in str(e):
                log_warning(f"OpenRouter rate limit hit, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                log_error(f"An error occurred with OpenRouter: {e}")
                break
    log_warning("Max retries reached. Could not complete the request.")
    return None

def generate_content_openai(client, model, prompt, temperature, max_retries=10, retry_delay=5):
    messages = [
        { "role": "user", "content": prompt}
    ]
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature)
            response = completion.choices[0].message.content
            return response  # If the request is successful, return the response
        except Exception as e:
            if 'RESOURCE_EXHAUSTED' in str(e):  # You can check for rate-limiting specific message in the error
                log_warning(f"LLM rate limit hit, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                log_error(f"An error occurred: {e}")
                break  # If it's not a rate-limiting error, stop retrying
    log_warning("Max retries reached. Could not complete the request.")
    
    return None  # Return None if the max retries are reached and the request was unsuccessful