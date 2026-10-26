import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://anyrouter.dev/api/v1", 
    api_key=os.getenv("API_KEY"),
)
MODEL = "nvidia/nemotron-3.5-bolt-e4b:free"


def _call(messages, temperature, top_p, max_tokens, json_mode):
    extra = {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"} 
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **extra,
    )
    return response.choices[0].message.content or "" 


def ask(prompt, system="You are a helpful assistant.", temperature=0,
        top_p=1, json_mode=False, max_tokens=1000):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(4): 
        try:
            answer = _call(messages, temperature, top_p, max_tokens, json_mode)
            time.sleep(2) 
            return answer
        except Exception as e:
            if json_mode and "response_format" in str(e).lower():
                print(" [note] JSON mode not supported here, using prompt-only JSON instead")
                json_mode = False
                messages[1]["content"] += "\n\nReturn only valid JSON."
                continue
            print(f" [retry {attempt + 1}] {str(e)[:100]}")
            time.sleep(10 * (attempt + 1))
    return "" 
