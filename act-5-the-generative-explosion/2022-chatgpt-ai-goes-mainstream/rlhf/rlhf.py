'''
RLHF
- we'll use ollama to generate multiple resposes for same prompt
- basic ranking system to rank responses
'''

# REWARD MODEL

# prompt + responses

import requests

def ask(prompt):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

# print("\n", ask("Explain gravity in one sentence"))
# print("\n", ask("Explain gravity in one sentence"))
# print("\n", ask("Explain gravity in one sentence"))

# different prompts to get different responses
prompts = [
    "Explain gravity in one sentence.",
    "Explain gravity in one sentence. Be very technical and use jargon.",
    "Explain gravity in one sentence. Respond casually like you're texting a friend.",
    "Explain gravity. Give a very long detailed explanation.",
    "Explain gravity. Keep it under 5 words.",
]

responses = []
for prompt in prompts:
    response = ask(prompt)
    responses.append(response)

# reward function
# response -> score(0-1)
def score_response(response):
    score = 0
    
    length = len(response)
    if 50 <= length <= 200:
        score += 0.5    # sweet spot
    elif 20 <= length < 50:
        score += 0.3    # a bit short
    elif 200 < length <= 400:
        score += 0.2    # getting verbose
    else:
        score += 0.05   # way too short or way too long
    
    sent_count = response.count(".")
    if sent_count == 1:
        score += 0.5    # perfect — followed "one sentence" instruction
    elif sent_count == 2:
        score += 0.3    # close
    elif sent_count <= 4:
        score += 0.1    # too many
    else:
        score += 0.01   # way too many

    return score

# score responses
scored_responses = []
for r in responses:
    s = score_response(r)
    scored_responses.append((r, s))
    print(f"Score: {s:.2f} | Response: {r[:60]}...")