# ChatGPT — AI Goes Mainstream (November 2022)

## GPT-3 is unpredictable

GPT-3 is powerful — it can write essays, translate, code, answer questions. But it's a wild animal. Ask "What's the capital of France?" and sometimes you get "Paris." Other times a 500-word essay about French history that never answers the question. Or worse — it makes up facts confidently, generates toxic content, or follows harmful instructions.

Why? Because GPT-3 was trained to **predict the next token**. Not to be helpful. Not to be safe. Just: given these words, what word comes next? It learned from the internet, and the internet sounds like *everything* — helpful, harmful, brilliant, stupid, all mixed together.

The question: how do you take a model that predicts text and turn it into one that is genuinely **helpful, harmless, and honest**?

This is the **alignment problem**.

## Can't we just fine-tune it?

Yes — and that's the first step. OpenAI hired people to write thousands of ideal conversations. A question paired with the perfect answer. Fine-tuned GPT-3 on these. This is called **Supervised Fine-Tuning (SFT)**.

It helped. The model got better at following instructions. But it wasn't enough.

SFT tells the model **what to say**. But it doesn't teach it **what humans prefer**. For many questions there are multiple valid answers — some clear, some verbose, some too technical. "Here's a recipe for chocolate cake" and "Chocolate cake was invented in 1764..." are both valid text. But one is what the user actually wanted.

You can't write an example for every possible question. And comparing "which answer is better" is much easier than writing the perfect answer from scratch. So instead of telling the model what to say — teach it what humans prefer.

## RLHF — Reinforcement Learning from Human Feedback

### Step 1: Supervised Fine-Tuning (SFT)

Collect high-quality conversations. Fine-tune base GPT-3 on them. Now you have a model that follows instructions reasonably well — but still not great. Sometimes too verbose, sometimes too brief, sometimes wrong.

### Step 2: Train a Reward Model

The new idea. Build a **separate model** that scores how good any response is.

How? Take a prompt, generate **multiple responses** from the SFT model, and have humans **rank** them:

```
Prompt: "Explain gravity in one sentence."

Response A: "Gravity attracts objects with mass toward each other."     → Clear, concise
Response B: "Gravity is a fundamental force involving spacetime..."     → Accurate but verbose
Response C: "idk google it lol"                                         → Useless

Human ranking: A > B > C
```

Collect thousands of these rankings. Train a model to predict the human ranking. Now the reward model can score any response:

```
Reward("Gravity attracts objects...")        → 0.92 (high)
Reward("Gravity is a fundamental force...") → 0.61 (medium)
Reward("idk google it lol")                 → 0.03 (low)
```

The reward model learned **what humans prefer** — not what's correct, but what's *preferred*. Clear, helpful, safe, appropriately detailed responses score high.

Why not just collect the best responses and train on those? Because ranking is **much easier** than writing. Comparing "A or B?" is quick. Writing the perfect response from scratch takes time and expertise. Rankings scale. Perfect examples don't.

### Step 3: Reinforcement Learning (the RL in RLHF)

Now use the reward model to **train the SFT model to produce higher-scoring responses**:

```
1. Take a prompt
2. SFT model generates a response
3. Reward model scores it: 0.65
4. Feedback: "not great, adjust weights to do better"
5. Model updates slightly
6. Repeat with a new prompt
```

Over thousands of prompts, the model shifts toward producing responses that score highly — responses that humans would prefer.

Same concept as training a dog: do something (action), get a treat or no treat (reward), learn to do more of what gets treats.

### Reward Hacking — the danger

What if the model finds a hack? What if it discovers the reward model gives high scores to responses that start with "Great question!" followed by a long, confident-sounding answer — even if the answer is wrong?

The model would **exploit** the reward model rather than actually being helpful. This is called **reward hacking**.

RLHF adds a constraint: the model can only make **small improvements**. No big jumps. It's penalized for being too different from the original SFT model.

The mechanism: **KL Divergence Penalty**. It measures how different the new model is from the original and adds that as a cost. Think of it as a leash — the model can improve, but it can't run wild.

## The Full Pipeline

```
Step 1: SFT
  Base GPT-3 → train on ideal conversations → follows instructions

Step 2: Reward Model
  Generate multiple responses → humans rank → train scorer

Step 3: Reinforcement Learning
  Model generates → reward model scores → model improves toward higher scores
  KL penalty prevents reward hacking

Result: ChatGPT
```

Same base model. Same architecture. Same 175 billion parameters. The difference is alignment — the model now produces what humans prefer because that's what gets high reward scores.

November 30, 2022 — OpenAI wraps this in a chat interface. **100 million users in two months.** Fastest-growing consumer app in history.

## What we built

A simplified reward model demonstrating the core concept. Generated multiple responses to the same prompt, scored them on qualities humans care about:

```
Score: 0.80 — short, clear answer (followed instructions)
Score: 0.70 — good but technical
Score: 0.30 — casual, rambling
Score: 0.06 — way too long (ignored "one sentence" instruction)
```

The scores match what a human would rank. A real reward model is a full neural network trained on hundreds of thousands of human rankings — but the concept is identical: response in, score out.

---

## Short Notes

**Previous problem:** GPT-3 was powerful but unpredictable — it could be helpful, harmful, verbose, terse, correct, or confidently wrong. It was trained to predict text, not to be useful.

**Solution this provided:** RLHF aligned the model with human preferences through three steps: SFT (follow instructions), reward model (learn what humans prefer), reinforcement learning (improve toward higher preference scores). This turned a text predictor into an assistant. The alignment problem didn't require a new architecture — just a new training process on top of the same model.