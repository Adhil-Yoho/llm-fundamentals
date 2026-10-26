"""
Task 4 - Prompt engineering toolkit.
Run: python task4_prompt_toolkit.py
"""
import json
from typing import List
from pydantic import BaseModel, ValidationError
from client import ask




tickets = json.load(open("data/golden_tickets.json", encoding="utf-8"))
resumes = json.load(open("data/resumes.json", encoding="utf-8"))


print("PART 1: Classification prompts on the golden set (15 tickets)")
prompt_files = {
    "v1": "prompts/v1.txt",
    "v2": "prompts/v2.txt",
    "v3": "prompts/v3.txt",
}

summary = []
for version, path in prompt_files.items():
    template = open(path, encoding="utf-8").read()
    correct = 0
    for t in tickets:
        reply = ask(template.replace("{ticket}", t["ticket"]), temperature=0).lower()
        if t["category"].lower() in reply and t["priority"].lower() in reply:
            correct += 1
    summary.append(f"{version}: {correct}/15 ({correct / 15:.0%})")
print(" | ".join(summary))


print("\nPART 2: Chainof thought vs plain")
problems = [
    ("A train goes 60 km/h for 2.5 hours, then 40 km/h for 1.5 hours. Total distance in km?", 210),
    ("Ravi has 3 boxes with 12 pencils each. He gives away 7 and buys 2 packs of 5. How many pencils now?", 39),
    ("A shirt costs 800 rupees. It gets 25% off, then 5% tax is added on the new price. Final price?", 630),
    ("Meena is 4 years older than twice Kavi's age. Kavi is 11. How old is Meena?", 26),
    ("A tank has 240 litres. It leaks 8 litres per hour for 9 hours, then 50 litres are added. Litres now?", 218),
]
plain_ok = cot_ok = 0
for question, answer in problems:
    plain = ask(question + " Reply with only the final number.", temperature=0)
    cot = ask(question + " Think step by step, then end with 'Final answer: <number>'.",
              temperature=0)
    if str(answer) in plain.replace(",", ""):
        plain_ok += 1
    if str(answer) in cot.replace(",", "")[-100:]:
        cot_ok += 1
print(f"Without 'think step by step': {plain_ok}/5 correct")
print(f"With 'think step by step': {cot_ok}/5 correct")


print("\nPART 3: Creative prompt, 5 runs each")
creative = "Write a one-line slogan for a small tea shop."

for temp in [0, 0.7, 1.2]:
    outputs = [ask(creative, temperature=temp) for _ in range(5)]
    print(f"\nTemperature {temp} -> {len(set(outputs))} different answers out of 5")
    for o in outputs:
        print(" -", o)

for top_p in [0.3, 0.95]:
    outputs = [ask(creative, temperature=1.0, top_p=top_p) for _ in range(5)]
    print(f"\nTop-p {top_p} (temperature 1.0) -> {len(set(outputs))} different answers out of 5")
    for o in outputs:
        print(" -", o)


class Resume(BaseModel):
    name: str
    skills: List[str]
    years_experience: float

def clean_json(raw):
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return raw.strip()
    return raw[start:end + 1].strip()

def extract(text, json_mode):
    prompt = ("Extract name, skills (a list) and years_experience (a number) "
              "from this resume paragraph as JSON.\n\n" + text)
    raw = ask(prompt, temperature=0, json_mode=json_mode)
    try:
        Resume.model_validate_json(clean_json(raw)) 
        return True, raw
    except ValidationError:
        return False, raw


print("\nPART 4: Resume extraction")
for json_mode in [False, True]:
    invalid = 0
    for r in resumes:
        ok, raw = extract(r["text"], json_mode)
        if not ok:
            invalid += 1
            print(" invalid output ->", raw[:80].replace("\n", " "))
    label = "JSON mode" if json_mode else "Plain "
    print(f"{label}: {invalid}/5 invalid")


print("\nPART 5: Bad prompts vs improved prompts")
text = ("Our new phone has a 6.5 inch screen, a 5000 mAh battery that lasts two days, "
        "and a 50 MP camera. It costs 15,000 rupees.")
pairs = [
    ("Tell me about Python",
     "You are a teacher. In 3 short bullet points, explain to a beginner what Python is "
     "used for in data analytics. Max 50 words."),
    ("Summarise this: " + text,
     "Summarise the product text below in 2 sentences for a shopper. "
     "Mention price and battery.\n\nText: " + text),
    ("Write an email",
     "Write a polite 4-line email to a customer saying their refund of 500 rupees "
     "will arrive in 3 working days. End with the name 'Support Team'."),
]
for bad, good in pairs:
    print("\nBAD PROMPT :", bad[:70])
    print("BAD ANSWER :", ask(bad).replace("\n", " ")[:250])
    print("GOOD PROMPT:", good[:70], "...")
    print("GOOD ANSWER:", ask(good).replace("\n", " ")[:250])
