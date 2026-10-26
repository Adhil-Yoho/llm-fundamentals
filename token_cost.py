import json
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

sentence = "The quick brown fox jumps over the lazy dog."

paragraph = (
    "Large language models read text as small pieces called tokens. A token can be a "
    "whole word, part of a word, or even a single symbol. The model never sees letters "
    "directly, it only sees numbers that stand for these tokens. This is why the price "
    "of using a model depends on the number of tokens and not on the number of words."
)

page = " ".join([paragraph] * 6)

json_text = json.dumps({
    "ticket_id": 1021,
    "customer": {"name": "Asha", "plan": "premium"},
    "issue": "Charged twice this month",
    "priority": "high",
    "tags": ["billing", "refund", "urgent"],
}, indent=2)

tamil = "நான் தினமும் காலையில் தேநீர் குடிப்பேன்."

texts = {
    "Sentence": sentence,
    "Paragraph": paragraph,
    "Page": page,
    "JSON": json_text,
    "Tamil sentence": tamil,
    "Big book (100 pages)": page * 100, 
}

print("TOKEN TABLE")
print(f"{'Text':<22}{'Words':>8}{'Tokens':>10}{'Words/Token':>14}")
token_counts = {}
for name, text in texts.items():
    words = len(text.split())
    tokens = len(enc.encode(text))
    token_counts[name] = tokens
    print(f"{name:<22}{words:>8}{tokens:>10}{words / tokens:>14.2f}")

print("\nCONTEXT WINDOW CHECK")
print(f"{'Text':<22}{'Fits 8k?':>12}{'Fits 128k?':>14}")
for name, tokens in token_counts.items():
    fits_8k = "Yes" if tokens <= 8_000 else "No"
    fits_128k = "Yes" if tokens <= 128_000 else "No"
    print(f"{name:<22}{fits_8k:>12}{fits_128k:>14}")


def estimate_cost(input_tokens, output_tokens, price_in, price_out):
    cost_in = input_tokens / 1_000_000 * price_in
    cost_out = output_tokens / 1_000_000 * price_out
    return cost_in + cost_out


models = {
    "Small model (example 0.15 in / 0.60 out)": (0.15, 0.60),
    "Large model (example 2.50 in / 10.00 out)": (2.50, 10.00),
}

chats_per_day = 1000
days = 30
print("\nMONTHLY COST (1000 chats/day, 1500 in + 400 out tokens, 30 days)")
for name, (p_in, p_out) in models.items():
    one_chat = estimate_cost(1500, 400, p_in, p_out)
    monthly = one_chat * chats_per_day * days
    print(f"{name}: per chat ${one_chat:.5f} | per month ${monthly:,.2f}")
