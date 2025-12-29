import time
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

    if model.startswith(("gemini", "gemma")):
        return generate_content_gemini(client, model, prompt, temperature, max_retries, retry_delay)
    else:
        return generate_content_huggingface(client, model, prompt, temperature, max_retries, retry_delay)    

def generate_content_gemini(client, model, prompt, temperature, max_retries=10, retry_delay=5):
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