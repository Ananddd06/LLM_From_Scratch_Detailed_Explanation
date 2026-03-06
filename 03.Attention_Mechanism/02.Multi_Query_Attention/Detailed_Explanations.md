```markdown
### 🧠 Multi-Query Attention (MQA)

Multi-Query Attention (MQA) is an **efficient variant of the Transformer attention mechanism** that reduces **memory usage and inference cost** in large language models.

It modifies the traditional **Multi-Head Attention (MHA)** by sharing **Keys and Values across heads**.

🚀 Many modern LLMs use this idea, including:

- :contentReference[oaicite:0]{index=0}  
- :contentReference[oaicite:1]{index=1}  
- :contentReference[oaicite:2]{index=2}  
- :contentReference[oaicite:3]{index=3}  

---

## 📦 Input to the Attention Layer

The input to attention is a tensor of token embeddings.

$$
X \in \mathbb{R}^{B \times T \times C}
$$

Where:

| Symbol | Meaning |
|------|------|
| **B** | Batch size |
| **T** | Sequence length |
| **C** | Embedding dimension |

Example:

```

B = 2
T = 1024
C = 4096

```

Tensor shape:

```

(2, 1024, 4096)

```

Each token embedding contains the **semantic representation of the token**.

---

## 🔧 Step 1 — Creating Query, Key, and Value

The first step in attention is projecting the input embeddings into three different vectors.

- 🔍 **Query (Q)** → what the token is looking for  
- 🏷 **Key (K)** → what information the token contains  
- 📦 **Value (V)** → the actual information passed forward  

Projection equations:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

Where the projection matrices are:

$$
W_Q \in \mathbb{R}^{C \times C}
$$

$$
W_K \in \mathbb{R}^{C \times C}
$$

$$
W_V \in \mathbb{R}^{C \times C}
$$

Resulting shapes:

$$
Q \in \mathbb{R}^{B \times T \times C}
$$

$$
K \in \mathbb{R}^{B \times T \times C}
$$

$$
V \in \mathbb{R}^{B \times T \times C}
$$

---

# 🧩 Standard Multi-Head Attention (MHA)

Before understanding MQA, we must understand how **standard multi-head attention works**.

---

## 🔹 Splitting into Multiple Heads

Instead of performing attention once, the model splits the embedding into **multiple heads**.

If the model has **H heads**, then each head dimension is:

$$
D = \frac{C}{H}
$$

Example:

```

Embedding dimension C = 4096
Heads H = 32

```

$$
D = 4096 / 32 = 128
$$

Now we reshape tensors.

Queries:

$$
Q \rightarrow (B, H, T, D)
$$

Keys:

$$
K \rightarrow (B, H, T, D)
$$

Values:

$$
V \rightarrow (B, H, T, D)
$$

Each head now learns **different relationships between tokens**.

---

## ⚙️ Step 2 — Attention Score Calculation

Each query is compared with all keys using **matrix multiplication**.

$$
S = QK^T
$$

Tensor shapes:

$$
Q \in \mathbb{R}^{B \times H \times T \times D}
$$

$$
K^T \in \mathbb{R}^{B \times H \times D \times T}
$$

Matrix multiplication:

$$
(B,H,T,D) \times (B,H,D,T)
$$

Result:

$$
S \in \mathbb{R}^{B \times H \times T \times T}
$$

This matrix tells **how strongly each token attends to other tokens**.

---

## ⚖️ Step 3 — Scaling

Large dot-products can destabilize training.

So we scale the scores:

$$
S = \frac{QK^T}{\sqrt{D}}
$$

Where:

$$
D = \text{head dimension}
$$

---

## 🔄 Step 4 — Softmax

Softmax converts the scores into probabilities.

$$
A = \text{softmax}(S)
$$

Shape:

$$
A \in \mathbb{R}^{B \times H \times T \times T}
$$

Each row now sums to **1**, representing attention distribution.

---

## 📊 Step 5 — Weighted Value Aggregation

Now the attention weights combine with the values.

$$
O = AV
$$

Shapes:

$$
(B,H,T,T) \times (B,H,T,D)
$$

Result:

$$
O \in \mathbb{R}^{B \times H \times T \times D}
$$

Each token now contains a **weighted mixture of other tokens**.

---

## 🔗 Step 6 — Concatenating Heads

All heads are merged together.

$$
(B,H,T,D) \rightarrow (B,T,H \times D)
$$

Since:

$$
H \times D = C
$$

Final output shape:

$$
(B,T,C)
$$

Then a final projection is applied:

$$
Y = OW_O
$$

Where:

$$
W_O \in \mathbb{R}^{C \times C}
$$

---

# ⚠️ Problem with Multi-Head Attention

The biggest problem is **KV cache memory usage during inference**.

KV cache stores **keys and values for every token**.

For MHA:

$$
K \in \mathbb{R}^{B \times H \times T \times D}
$$

$$
V \in \mathbb{R}^{B \times H \times T \times D}
$$

Memory required:

$$
2 \times B \times H \times T \times D
$$

Example:

```

Heads = 32
Sequence = 8192
Head_dim = 128

```

KV cache becomes **very large**.

---

# 🚀 Multi-Query Attention (MQA)

Multi-Query Attention solves this problem.

Instead of giving **each head its own Key and Value**, MQA shares them.

---

## 🧩 Architecture

Standard MHA:

```

Q heads = H
K heads = H
V heads = H

```

MQA:

```

Q heads = H
K heads = 1
V heads = 1

```

So all query heads read from **one shared memory**.

---

## 🔧 MQA Projection

Query projection:

$$
Q = XW_Q
$$

Where:

$$
W_Q \in \mathbb{R}^{C \times C}
$$

Key projection:

$$
K = XW_K
$$

Where:

$$
W_K \in \mathbb{R}^{C \times D}
$$

Value projection:

$$
V = XW_V
$$

Where:

$$
W_V \in \mathbb{R}^{C \times D}
$$

Resulting shapes:

$$
Q \in \mathbb{R}^{B \times T \times C}
$$

$$
K \in \mathbb{R}^{B \times T \times D}
$$

$$
V \in \mathbb{R}^{B \times T \times D}
$$

---

## 🔄 Expanding Keys and Values

Queries are split into heads:

$$
Q \rightarrow (B,H,T,D)
$$

But Keys and Values remain:

$$
K \rightarrow (B,T,D)
$$

$$
V \rightarrow (B,T,D)
$$

We expand them:

$$
K \rightarrow (B,1,T,D)
$$

$$
V \rightarrow (B,1,T,D)
$$

Broadcast:

$$
K \rightarrow (B,H,T,D)
$$

$$
V \rightarrow (B,H,T,D)
$$

⚠️ Important: This **does not duplicate memory** — it just reuses the same tensor.

---

## ⚙️ Attention Computation in MQA

The attention computation remains the same.

Score calculation:

$$
S = QK^T
$$

Scaling:

$$
S = \frac{QK^T}{\sqrt{D}}
$$

Softmax:

$$
A = \text{softmax}(S)
$$

Output:

$$
O = AV
$$

Merge heads:

$$
(B,H,T,D) \rightarrow (B,T,C)
$$

---

# 💾 KV Cache Reduction

Standard MHA KV cache:

$$
K,V \in \mathbb{R}^{B \times H \times T \times D}
$$

Memory:

$$
2HTD
$$

MQA KV cache:

$$
K,V \in \mathbb{R}^{B \times T \times D}
$$

Memory:

$$
2TD
$$

Reduction:

$$
\text{Memory reduction} \approx H \times
$$

Example:

```

H = 32

```

Memory becomes **32× smaller**.

---

# 🧠 Intuition

Think of it like this:

Standard MHA:

```

32 heads → 32 separate memories

```

MQA:

```

32 heads → 1 shared memory

```

All heads **read from the same knowledge store**.

---

# 📊 Comparison

| Feature | Multi-Head Attention | Multi-Query Attention |
|------|------|------|
Query heads | H | H |
Key heads | H | 1 |
Value heads | H | 1 |
KV cache | Large | Small |
Inference speed | Slower | Faster |
Memory | High | Low |

---

# 🏁 Summary

Multi-Query Attention improves Transformer efficiency by **sharing keys and values across heads**.

Key advantages:

✅ Dramatically smaller KV cache  
✅ Faster inference  
✅ Better scalability for long contexts  
✅ Widely used in modern LLMs  

Because of these advantages, MQA has become a **core optimization in modern large language models**.
```
