import requests

# Set the API URL and your API key (replace with actual API URL and key)
API_URL = "https://huggingface.co/settings/tokens"  # Replace with your API URL
API_KEY = "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"  # Replace with your actual API key

# Define headers for authorization
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


# Function to call the API with the prompt
def call_llama_api(prompt, max_tokens=100):
    """
    Calls the LLaMA API to generate a response for the given prompt.
    """
    data = {
        "model": "llama-8b",  # Model name, adjust if necessary
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    # Sending a POST request to the API
    response = requests.post(API_URL, headers=HEADERS, json=data)

    # Check if the response was successful
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("text", "")
    else:
        return f"Error {response.status_code}: {response.text}"


# Example usage
prompt = "Explain the concept of reinforcement learning in autonomous vehicles."
response = call_llama_api(prompt)

# Print the response
print(response)
