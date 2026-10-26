| Test | Prompt | Result | Rating |
|---|---|---|---|
| Summarisation | Summarise in one sentence:
Priya ordered a blue jacket from the online store on  | Priya, who ordered a blue jacket on March 3rd, received a late and damaged item on March 12th, and after two unanswered support calls, is now requesti | 5 |
| Classification | Is this customer message Positive, Negative or Neutral? Reply with one word.
Pri | Negative | 5 |
| Extraction | From the text, extract the customer name, product and delay in days:
Priya order | * **Customer Name:** Priya * **Product:** blue jacket * **Delay in Days:** 9 | 5 |
| Translation | Translate to Tamil: 'Your order will arrive tomorrow morning.' | Here are a few ways to translate "Your order will arrive tomorrow morning" into Tamil, depending on the level of formality you prefer:  **1. Standard/ | 5 |
| Code generation | Write a simple Python function that checks if a word is a palindrome. | Here is a simple and efficient Python function to check if a word is a palindrome:  ```python def is_palindrome(word):   """   Checks if a given word  | 5 |
| Knowledge cutoff | What was the biggest news story in the world this week? Give the exact date. | As an AI, I don't have real-time access to the absolute "biggest" news story globally for this specific week, as what is considered "biggest" is subje | 5 |
| Hallucination (normal) | Summarise the 2021 paper 'Quantum Banana Attention' by Dr. Zorvan Kelpington. | I am sorry, but I could not find any record of a 2021 paper titled **"Quantum Banana Attention"** by a **Dr. Zorvan Kelpington** in major academic dat | 5 |
| Hallucination (with 'say I don't know') | Summarise the 2021 paper 'Quantum Banana Attention' by Dr. Zorvan Kelpington. If | I don't know. | 5 |
| Long context (fact in middle) | [~20k tokens of filler with a secret code in the middle] What is the secret code | The secret code mentioned in the text is **PINEAPPLE-42**. | 5 |
| Multi-step arithmetic | A pen costs 12 rupees. Ravi buys 7 pens and gets a 10% discount on the total. He | Here's how to solve the problem step-by-step:  **1. Calculate the total cost before the discount:** * Cost per pen: 12 rupees * Number of pens: 7 * To | 5 |
| Logic puzzle | Anu is taller than Bala. Bala is taller than Charu. Dev is shorter than Charu. W | Here's how to figure out the order:  1. **Anu is taller than Bala:** Anu > Bala 2. **Bala is taller than Charu:** Bala > Charu 3. **Dev is shorter tha | 5 |

## 5 Safeguards
1. Ground answers in our own documents (RAG) and ask the model to cite them.
2. Add 'say I don't know if unsure' in the system prompt and show a confidence note.
3. Never let the model do money or date maths alone: use a calculator or code tool.
4. Validate outputs (JSON schema, allowed values) and retry or fall back if invalid.
5. Keep a human in the loop for refunds, legal or medical topics, and log every answer.
