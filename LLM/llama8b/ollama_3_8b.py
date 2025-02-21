import ollama

response = ollama.chat(
    model="llama2",
    messages=[
        {
            "role": "user",
            "content": "Tell me the importance of reinforcement learning.",
        },
    ],
)
print(response["message"]["content"])