# LLM Fundamentals Lab: Tokens, Transformers and Prompt Engineering

A hands-on project that measures token cost, tests what LLMs can and cannot do, explains the transformer with working code, and builds a small prompt-engineering toolkit.

- **Model used:** `nvidia/nemotron-3.5-bolt-e4b:free` (through an OpenAI-compatible API)
- **Language:** Python

## Project structure

```
llm-fundamentals/
├── client.py # one function (ask) that calls the model
├── token_cost.py # Task 1: tokens, context windows, cost
├── limitations.py # Task 2: capabilities and limitations lab
├── t3_transformers.py # Task 3: attention, positional encoding, BERT, GPT-2
├── t4_prompt_toolkit.py # Task 4: prompt engineering toolkit
├── data/
│ ├── golden_tickets.json # 15 support tickets with correct answers
│ └── resumes.json # 5 resume paragraphs with expected JSON
├── prompts/
│ ├── v1.txt # basic prompt
│ ├── v2.txt # Role / Task / Constraints / Output format
│ └── v3.txt # v2 + 3 few-shot examples
├── outputs/
│ ├── task2_results.md
│ └── positional_encoding.png
├── requirements.txt
├── .env.example
└── .gitignore
```

## How to run

1. Install the libraries: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your key: `API_KEY=your-key-here`
3. Run each task from the project folder

---

## Task 1 - Models, Tokens and Cost

### 1. Model family comparison

| Family | Owner | Weights | Context window* | Strengths |
|---|---|---|---|---|
| GPT | OpenAI | Closed | ~128k to 1M (depends on model) | Strong all-rounder, tools, vision |
| Claude | Anthropic | Closed | ~200k, up to 1M on some models | Long documents, coding, careful answers |
| Llama | Meta | Open weights | ~128k and above | Free to self-host, private data, fine-tuning |
| Gemini | Google | Closed | ~1M and above | Very long context, multimodal |

Sources:
- OpenAI: https://platform.openai.com/docs/models
- Anthropic: https://docs.claude.com/en/docs/about-claude/models
- Meta Llama: https://www.llama.com/
- Google Gemini: https://ai.google.dev/gemini-api/docs/models

### 2. Token table (tiktoken, `o200k_base`)

| Text | Words | Tokens | Words/Token |
|---|---|---|---|
| Sentence | 9 | 10 | 0.90 |
| Paragraph | 63 | 70 | 0.90 |
| Page | 378 | 420 | 0.90 |
| JSON | 24 | 70 | 0.34 |
| Tamil sentence | 5 | 13 | 0.38 |
| Big book (100 pages) | 37,701 | 42,000 | 0.90 |

**What I learned:** plain English is about 0.9 words per token. JSON and Tamil use far more tokens per word (about 0.34 and 0.38), so they cost more and fill the context window faster.

### 3. Context window check

| Text | Fits 8k? | Fits 128k? |
|---|---|---|
| Sentence, Paragraph, Page, JSON, Tamil | Yes | Yes |
| Big book (100 pages) | No | Yes |

### 4. Cost calculation

`estimate_cost(input_tokens, output_tokens, price_in, price_out)` uses prices per 1M tokens.
Scenario: 1000 chats per day, 1500 input + 400 output tokens each, 30 days.

| Model (example prices per 1M tokens) | Per chat | Per month |
|---|---|---|
| Small model (0.15 in / 0.60 out) | $0.00047 | $13.95 |
| Large model (2.50 in / 10.00 out) | $0.00775 | $232.50 |

The prices are example values to show the calculation. Real prices are on each provider's pricing page.

### 5. Recommendation

1. Public support bot: a small closed API model such as GPT-4o-mini or Claude Haiku.
2. It is cheap per chat, fast, and the provider handles scaling and safety.
3. Closed APIs also give quick upgrades and good multilingual support.
4. Private on-premise tool: Llama (open weights) running on our own servers.
5. Data never leaves the company, there is no per-token bill, and it can be fine-tuned.

---

## Task 2 - Capabilities and Limitations Lab

### Results table

| Test | Prompt | Result | Rating |
|---|---|---|---|
| Summarisation | Summarise in one sentence (Priya's jacket complaint) | Correct one-sentence summary with dates and both demands | 5 |
| Classification | Positive, Negative or Neutral? One word | "Negative" | 5 |
| Extraction | Extract customer name, product, delay in days | Priya / blue jacket / 9 days | 5 |
| Translation | Translate to Tamil: "Your order will arrive tomorrow morning." | Gave 3 correct Tamil versions with explanation | 5 |
| Code generation | Python function to check a palindrome | Working function, examples and a simpler version | 5 |
| Knowledge cutoff | Biggest news story this week, with exact date | Said it has no real-time access and asked for the date range | 5 |
| Hallucination (normal) | Summarise the 2021 paper "Quantum Banana Attention" by Dr. Zorvan Kelpington | Did not invent a summary; said it could not find the paper and guessed it may be fictional | 5 |
| Hallucination (with "say I don't know") | Same prompt + "if unsure, say I don't know" | "I don't know." | 5 |
| Long context (fact in the middle) | ~12k tokens of filler with a secret code in the middle | Found PINEAPPLE-42 correctly | 5 |
| Multi-step arithmetic | 7 pens at 12 rupees, 10% off, pay 100 | 24.40 rupees (matches my hand answer 24.4) | 5 |
| Logic puzzle | Who is second tallest? | Bala (matches my hand answer) | 5 |

**Notes**
- Hallucination: this model was already careful on the normal prompt. The "say I don't know" line made the answer shorter and more direct.
- Both reasoning answers were checked by hand: 7 x 12 = 84, minus 10% = 75.6, change = 24.4. And Anu > Bala > Charu > Dev.
- Long context: the model found the fact in the middle, but this was an easy test (repeated filler). Real documents are harder.

### 5 safeguards for a business application

1. Ground answers in our own documents (RAG) and ask the model to cite them.
2. Add "say I don't know if unsure" in the system prompt and show a confidence note.
3. Never let the model do money or date maths alone: use a calculator or code tool.
4. Validate outputs (JSON schema, allowed values) and retry or fall back if invalid.
5. Keep a human in the loop for refunds, legal or medical topics, and log every answer.

---

## Task 3 - Transformer Concepts in Code

Everything is plain NumPy (4 tokens, 8 dimensions, `np.random.seed(7)`).

### Single-head self-attention (4x4 weights)

```
[[0.003 0. 0.001 0.996]
 [0.767 0. 0. 0.232]
 [0.713 0.284 0.003 0. ]
 [0.1 0.9 0. 0. ]]
Row sums: [1. 1. 1. 1.]
Verified: each row sums to 1.0
```

Steps: Q, K, V = X @ Wq, Wk, Wv, then scores = Q @ K.T / sqrt(d), then row-wise softmax, then output = weights @ V.

### Two heads, concatenate, output projection

- Each head uses 4 of the 8 dimensions and has its own 4x4 attention matrix.
- Concatenated shape: (4, 8). After the output projection `Wo`: (4, 8).

### Sinusoidal positional encoding (10 positions x 8 dimensions)

Saved plot: `outputs/positional_encoding.png`. Position 0 is `[0, 1, 0, 1, 0, 1, 0, 1]` (sin = 0, cos = 1), and the first dimensions change fast while the last ones change slowly.

### Hugging Face examples

- **BERT fill-mask:** "The capital of France is [MASK]." gave `paris` (0.417), `lille` (0.071), `lyon` (0.063).
- **GPT-2 text-generation:** "Machine learning is" continued with "only a few years old, and it's all about how it's implemented. In this article,".

### Encoder-only vs decoder-only, and why transformers replaced RNNs

1. Encoder-only models (like BERT) read the whole sentence both ways, so they understand text well.
2. They are used for classification, search and fill-in-the-blank tasks.
3. Decoder-only models (like GPT) read left to right and predict the next token.
4. They are used for writing, chat and code, which is why today's LLMs are decoders.
5. RNNs read one word at a time, so they are slow to train and forget long-range context.
6. Transformers use attention to see all words at once, train in parallel and handle long text better.

---

## Task 4 - Prompt Engineering Toolkit

### Golden set
- `data/golden_tickets.json`: 15 tickets (5 Billing, 5 Technical, 5 Account) with a priority each.
- `data/resumes.json`: 5 resume paragraphs with the expected JSON.

### Classification prompts (temperature 0)
- **v1:** basic prompt
- **v2:** Role / Task / Constraints / Output format
- **v3:** v2 + 3 few-shot examples

| Version | Correct | Accuracy |
|---|---|---|
| v1 | 7/15 | 47% |
| v2 | 12/15 | 80% |
| v3 | 10/15 | 67% |

A ticket counts as correct only if both the category and the priority appear in the reply.
**Takeaway:** adding structure (v2) gave the biggest jump, from 47% to 80%. The few-shot version did not beat v2 here; with only 15 tickets and a simple scoring rule, small changes move the score a lot.

### Chain-of-thought (5 word problems)

| Prompt | Correct |
|---|---|
| Without "think step by step" (final number only) | 0/5 |
| With "think step by step" | 5/5 |

### Temperature and top-p (slogan prompt, 5 runs each)

| Setting | Different answers out of 5 |
|---|---|
| Temperature 0 | 1 |
| Temperature 0.7 | 1 |
| Temperature 1.2 | 1 |
| Top-p 0.3 | 1 |
| Top-p 0.95 | 1 |

Each setting gave one answer repeated five times (for example "Your cozy corner for a perfect cup." at temperature 0 and "Sip, unwind, and savor the moment." at top-p 0.95). I expected more variety at higher temperature. The free model or its API may ignore sampling settings or reuse a stored response for identical requests, so this test did not show the normal effect.

### Structured extraction with Pydantic

| Mode | Invalid (before fix) | Invalid (after fix) |
|---|---|---|
| Plain prompt | 5/5 | 0/5 |
| JSON mode | 5/5 | 0/5 |

**Takeaway:** never trust raw model output. Clean it and validate it (here with Pydantic) before using it.

### Bad prompts rewritten

| Bad prompt | Problem | Improved prompt | Result |
|---|---|---|---|
| "Tell me about Python" | Too vague, long answer | Role + audience + "3 short bullets, max 50 words" | 3 short, useful bullets |
| "Summarise this: ..." | No length or focus | "2 sentences for a shopper, mention price and battery" | 2 focused sentences |
| "Write an email" | Model asked questions back | Clear topic, tone, 4 lines, sign-off "Support Team" | A complete 4-line email |

---

## Key learnings

- Tokens, not words, decide cost and context limits, and some languages and formats use many more tokens
- Clear structure in a prompt (role, task, constraints, output format) improved accuracy the most
- Asking the model to think step by step fixed all five maths problems
- Always clean and validate model output (for example with Pydantic) before using it in a real application. A small cleaning step fixed all 5 failed extractions
- Always test settings like temperature on the actual API, since some providers do not apply them
