from client import ask

results = []


def run_test(test, prompt, system="You are a helpful assistant.", show_prompt=None):
    print("\n" + "=" * 60)
    print("TEST:", test)
    print("PROMPT:", show_prompt or prompt)
    answer = ask(prompt, system=system)
    print("ANSWER:", answer)
    rating = input("Rate this answer 1-5: ").strip()
    short = answer.replace("\n", " ")[:150]
    results.append((test, (show_prompt or prompt)[:80], short, rating))
    return answer


sample = ("Priya ordered a blue jacket from the online store on 3 March. It arrived on 12 March, "
          "nine days late, and the zip was broken. She called support twice but nobody called back. "
          "She now wants a replacement jacket or her money back.")

run_test("Summarisation", f"Summarise in one sentence:\n{sample}")
run_test("Classification",
         f"Is this customer message Positive, Negative or Neutral? Reply with one word.\n{sample}")
run_test("Extraction",
         f"From the text, extract the customer name, product and delay in days:\n{sample}")
run_test("Translation", "Translate to Tamil: 'Your order will arrive tomorrow morning.'")
run_test("Code generation",
         "Write a simple Python function that checks if a word is a palindrome.")

run_test("Knowledge cutoff",
         "What was the biggest news story in the world this week? Give the exact date.")

fake = "Summarise the 2021 paper 'Quantum Banana Attention' by Dr. Zorvan Kelpington."
run_test("Hallucination (normal)", fake)
run_test("Hallucination (with 'say I don't know')",
         fake + " If you are not sure this paper exists, say 'I don't know'.")

filler = "The weather in the city was calm and the market was busy. " * 1000
half = len(filler) // 2
long_text = filler[:half] + " The secret code is PINEAPPLE-42. " + filler[half:]
run_test("Long context (fact in middle)",
         long_text + "\n\nWhat is the secret code mentioned in the text above?",
         show_prompt="[~20k tokens of filler with a secret code in the middle] What is the secret code?")

run_test("Multi-step arithmetic",
         "A pen costs 12 rupees. Ravi buys 7 pens and gets a 10% discount on the total. "
         "He pays with a 100 rupee note. How much change does he get?")
print("CORRECT BY HAND: 7 x 12 = 84, minus 10% = 75.6, change = 24.4 rupees")

run_test("Logic puzzle",
         "Anu is taller than Bala. Bala is taller than Charu. Dev is shorter than Charu. "
         "Who is the second tallest?")
print("CORRECT BY HAND: Anu > Bala > Charu > Dev, so the second tallest is Bala")

safeguards = [
    "Ground answers in our own documents (RAG) and ask the model to cite them.",
    "Add 'say I don't know if unsure' in the system prompt and show a confidence note.",
    "Never let the model do money or date maths alone: use a calculator or code tool.",
    "Validate outputs (JSON schema, allowed values) and retry or fall back if invalid.",
    "Keep a human in the loop for refunds, legal or medical topics, and log every answer.",
]

with open(f"outputs/task2_results.md", "w", encoding="utf-8") as f:
    f.write("| Test | Prompt | Result | Rating |\n|---|---|---|---|\n")
    for test, prompt, result, rating in results:
        f.write(f"| {test} | {prompt.replace('|', '/')} | {result.replace('|', '/')} | {rating} |\n")
    f.write("\n## 5 Safeguards\n")
    for i, s in enumerate(safeguards, 1):
        f.write(f"{i}. {s}\n")

print("\nSaved task2_results.md")
print("\n5 SAFEGUARDS")
for i, s in enumerate(safeguards, 1):
    print(f"{i}. {s}")
