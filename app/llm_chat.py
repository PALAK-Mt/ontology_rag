# app/llm_chat.py
import os
import requests
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


API_URL = "https://router.huggingface.co/together/v1/chat/completions"


if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN environment variable not set.")

def chat_with_model(messages: list, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1") -> str:
    import os
    import requests

    API_URL = "https://router.huggingface.co/together/v1/chat/completions"
    HF_TOKEN = os.getenv("HF_TOKEN")

    if HF_TOKEN is None:
        raise RuntimeError("❌ HF_TOKEN environment variable not set.")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 800
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if not response.ok:
        print("❌ Request failed:")
        print("Status:", response.status_code)
        print("Response:", response.text)
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
