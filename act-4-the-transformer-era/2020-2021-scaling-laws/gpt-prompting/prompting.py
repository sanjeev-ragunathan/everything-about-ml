'''
We're gonna run local models in our system.
- we'll run llama3.2:1b using Ollama
'''

import requests

def ask(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False,
        }
    )
    return response.json()["response"]

# print(ask("What is 2 + 2?")) # testing



# SIMPLE PROMPTING TASKS

# # Zero-shot — just ask
# zero_shot = ask("Classify the sentiment of this review as Positive or Negative. Only respond with one word.\n\nReview: 'The food was cold and the waiter was rude.'\n\nSentiment:")

# # Few-shot — give examples first
# few_shot = ask("""Classify the sentiment of each review as Positive or Negative.

# Review: 'Best pizza I have ever had!' → Positive
# Review: 'Terrible service, never coming back' → Negative
# Review: 'The ambiance was lovely and food was decent' → Positive

# Review: 'The food was cold and the waiter was rude' →""")

# print(f"Zero-shot: {zero_shot}")
# print(f"Few-shot: {few_shot}")



# DIFFCULT PROMPTING TASKS

# # Zero-shot — word to emoji
# zero_emoji = ask("Convert this word to an emoji. Only respond with the emoji.\n\nWord: 'fire'\n\nEmoji:")

# # Few-shot — word to emoji with examples
# few_emoji = ask("""Convert words to emojis.

# sun → ☀️
# heart → ❤️
# rain → 🌧️
# cat → 🐱

# fire →""")

# print(f"Zero-shot emoji: {zero_emoji}")
# print(f"Few-shot emoji: {few_emoji}")



# Direct — no reasoning
direct = ask("What is 47 x 83? Only respond with the number.")

# Chain-of-thought — think step by step
cot = ask("What is 47 x 83? Think step by step, show your work, then give the final answer.")

print(f"Direct: {direct}")
print(f"Chain-of-thought: {cot}")