import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

np.random.seed(7)


def softmax(x):
    """Row-wise softmax: each row becomes positive numbers that add up to 1"""
    e = np.exp(x - x.max(axis=1, keepdims=True)) 
    return e / e.sum(axis=1, keepdims=True)


def self_attention(X, Wq, Wk, Wv):
    Q = X @ Wq 
    K = X @ Wk 
    V = X @ Wv 
    d = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d) 
    weights = softmax(scores) 
    output = weights @ V 
    return output, weights


tokens = 4 
d_model = 8 
X = np.random.randn(tokens, d_model)

Wq = np.random.randn(d_model, d_model)
Wk = np.random.randn(d_model, d_model)
Wv = np.random.randn(d_model, d_model)

out, weights = self_attention(X, Wq, Wk, Wv)
print("PART 1: Attention weights (4x4):")
print(np.round(weights, 3))
print("Row sums:", weights.sum(axis=1))
assert np.allclose(weights.sum(axis=1), 1.0), "rows must sum to 1"
print("Verified: each row sums to 1.0")
print("Output shape:", out.shape)

d_head = d_model // 2 
head_outputs = []
for h in range(2):
    Wq_h = np.random.randn(d_model, d_head)
    Wk_h = np.random.randn(d_model, d_head)
    Wv_h = np.random.randn(d_model, d_head)
    head_out, head_w = self_attention(X, Wq_h, Wk_h, Wv_h)
    print(f"\nHead {h + 1} attention weights:")
    print(np.round(head_w, 3))
    head_outputs.append(head_out)

concat = np.concatenate(head_outputs, axis=1) 
Wo = np.random.randn(d_model, d_model)
multi_out = concat @ Wo
print("\nPART 2: concatenated shape:", concat.shape)
print("Final multi-head output shape:", multi_out.shape)


def positional_encoding(positions, dims):
    pe = np.zeros((positions, dims))
    for pos in range(positions):
        for i in range(0, dims, 2):
            angle = pos / (10000 ** (i / dims))
            pe[pos, i] = np.sin(angle) 
            pe[pos, i + 1] = np.cos(angle) 
    return pe

pe = positional_encoding(10, 8)
print("\nPART 3: Positional encoding (10 positions x 8 dims):")
print(np.round(pe, 3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
im = ax[0].imshow(pe, cmap="RdBu", aspect="auto")
ax[0].set_title("Heatmap")
ax[0].set_xlabel("Dimension")
ax[0].set_ylabel("Position")
fig.colorbar(im, ax=ax[0])
for d in range(8):
    ax[1].plot(pe[:, d], label=f"dim {d}")
ax[1].set_title("Each dimension across positions")
ax[1].set_xlabel("Position")
ax[1].legend(fontsize=6)
plt.tight_layout()
plt.show
plt.savefig(f"outputs/positional_encoding.png")
print("Saved plot: positional_encoding.png")

from transformers import pipeline, set_seed

print("\nPART 4a: BERT fill-mask")
fill = pipeline("fill-mask", model="bert-base-uncased")
for r in fill("The capital of France is [MASK].")[:3]:
    print(f" {r['token_str']} (score {r['score']:.3f})")

print("\nPART 4b: GPT-2 text-generation")
set_seed(7)
gen = pipeline("text-generation", model="gpt2")
print(gen("Machine learning is", max_new_tokens=20, num_return_sequences=1)[0]["generated_text"])

