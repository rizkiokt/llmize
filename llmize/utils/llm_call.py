import time

def generate_content(client, model, prompt, max_retries=10, retry_delay=5):
    if model.startswith("gemini"):
        return generate_content_gemini(client, model, prompt, max_retries, retry_delay)
    else:
        return generate_content_huggingface(client, model, prompt, max_retries, retry_delay)    

def generate_content_gemini(client, model, prompt, max_retries=10, retry_delay=5):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text  # If the request is successful, return the response
        except Exception as e:
            if 'RESOURCE_EXHAUSTED' in str(e):  # You can check for rate-limiting specific message in the error
                print(f"Rate limit hit, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                print(f"An error occurred: {e}")
                break  # If it's not a rate-limiting error, stop retrying
    print("Max retries reached. Could not complete the request.")
    return None  # Return None if the max retries are reached and the request was unsuccessful

def generate_content_huggingface(client, model, prompt, max_retries=10, retry_delay=5):
    messages = [
        { "role": "user", "content": prompt}
    ]
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages)
            response = completion.choices[0].message.content
            return response  # If the request is successful, return the response
        except Exception as e:
            if 'RESOURCE_EXHAUSTED' in str(e):  # You can check for rate-limiting specific message in the error
                print(f"Rate limit hit, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                print(f"An error occurred: {e}")
                break  # If it's not a rate-limiting error, stop retrying
    print("Max retries reached. Could not complete the request.")
    return None  # Return None if the max retries are reached and the request was unsuccessful