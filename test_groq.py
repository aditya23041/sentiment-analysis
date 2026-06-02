import os
from openai import OpenAI

try:
    client = OpenAI(api_key="gsk_1234567890", base_url="https://invalid.groq.com/openai/v1", max_retries=0)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0
    )
    print(response)
except Exception as e:
    print(f"Exception: {type(e).__name__}: {str(e)}")
