### 🧠 Grouped Query Attention (GQA) — Detailed Explanation

Grouped Query Attention (GQA) is a modern attention mechanism designed to **reduce KV cache memory while maintaining model quality**.
It sits **between Multi-Head Attention (MHA) and Multi-Query Attention (MQA)**.

Many modern LLMs use it, including:

- Llama 3
- Gemma
- Gemini

---

# 1️⃣ Standard Multi-Head Attention (MHA)

## Idea

Each head has **its own Q, K, V projections**.

### Formula

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

Each head computes:

$$
Attention_i(Q,K,V) =
softmax\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i
$$

Final output:

$$
Concat(head_1,...,head_h)W_O
$$

---

## Architecture

Example with **8 heads**

```text
Q1 K1 V1
Q2 K2 V2
Q3 K3 V3
Q4 K4 V4
Q5 K5 V5
Q6 K6 V6
Q7 K7 V7
Q8 K8 V8
```

Each head has **separate K and V**.

---

## Problem of MHA

During inference we store **KV cache**.

Memory:

$$
KV = N \times H \times D
$$

Where

- (N) = sequence length
- (H) = number of heads
- (D) = head dimension

Example:

```text
N = 32k
H = 32
D = 128
```

KV cache becomes extremely large.

This slows inference.

---

# 2️⃣ Multi-Query Attention (MQA)

MQA was introduced to **reduce KV cache memory**.

Used by early large models like:

- PaLM

---

## Idea

All heads share **one single K and V**.

Architecture:

```text
Q1
Q2
Q3
Q4
Q5
Q6
Q7
Q8
      ↓
     K,V
```

Each head has its **own Q**, but **same K,V**.

---

## Formula

Queries remain head-specific:

$$
Q_i = XW_{Q_i}
$$

But

$$
K = XW_K
$$

$$
V = XW_V
$$

All heads attend to the **same keys and values**.

---

## Memory Improvement

KV cache becomes:

$$
KV = N \times D
$$

instead of

$$
N \times H \times D
$$

This reduces KV memory by **H times**.

Example:

```text
32 heads → 32× smaller KV cache
```

---

## Problem with MQA

MQA reduces memory **too aggressively**.

Since all heads share the same **K,V**, the model loses **head diversity**.

This causes:

❌ reduced model quality
❌ weaker representation power
❌ worse perplexity

Because heads cannot specialize.

---

# 3️⃣ Grouped Query Attention (GQA)

GQA is a **middle ground** between MHA and MQA.

Instead of **1 KV for all heads**, we create **groups of heads sharing KV**.

---

## Example

Suppose:

```text
num_heads = 8
num_kv_heads = 2
```

Grouping:

```text
Q1 Q2 Q3 Q4 → KV1
Q5 Q6 Q7 Q8 → KV2
```

So we have:

```text
8 query heads
2 KV heads
```

---

## Formula

Queries:

$$
Q_i = XW_{Q_i}
$$

Grouped keys and values:

$$
K_g = XW_{K_g}
$$

$$
V_g = XW_{V_g}
$$

Where (g) is the **group index**.

Each query head attends to its **group's KV pair**.

---

# 4️⃣ Architecture Comparison

## Multi-Head Attention

```text
Q1 K1 V1
Q2 K2 V2
Q3 K3 V3
Q4 K4 V4
Q5 K5 V5
Q6 K6 V6
Q7 K7 V7
Q8 K8 V8
```

---

## Multi-Query Attention

```text
Q1
Q2
Q3
Q4
Q5
Q6
Q7
Q8
      ↓
     K,V
```

---

## Grouped Query Attention

```text
Q1 Q2 Q3 Q4 → KV1
Q5 Q6 Q7 Q8 → KV2
```

Balanced design.

---

# 5️⃣ Memory Comparison

KV cache size:

### MHA

$$
KV = N \times H \times D
$$

---

### MQA

$$
KV = N \times D
$$

---

### GQA

$$
KV = N \times G \times D
$$

Where

- (G) = number of KV groups

---

Example:

```text
heads = 32
kv_heads = 8
```

Memory reduction:

```text
32 / 8 = 4× smaller KV cache
```

---

# 6️⃣ Why GQA Is Important

GQA balances:

```text
memory efficiency
+
model quality
```

Benefits:

✅ smaller KV cache
✅ faster inference
✅ better head diversity than MQA
✅ better GPU efficiency

---

# 7️⃣ Why Masking Works Well with GQA

Masking in transformers controls **which tokens can attend to which tokens**.

Causal mask:

$$
Mask(i,j)=
\begin{cases}
0 & j \le i \
-\infty & j > i
\end{cases}
$$

In GQA:

- **Queries remain per-head**
- Mask is applied to **attention scores**

So masking still works **exactly like MHA**.

No information leakage.

Because masking happens at:

$$
QK^T
$$

before softmax.

---

# 8️⃣ Why GQA Is Better for Memory

During inference we store **KV cache for every generated token**.

Memory per token:

### MHA

```text
KV for each head
```

Large memory.

---

### GQA

```text
KV only for groups
```

Much smaller memory.

This allows models to support:

```text
longer context
faster decoding
```

---

# 9️⃣ Limitation of MQA Solved by GQA

MQA limitation:

```text
all heads share same K,V
```

This reduces **attention diversity**.

Different heads should focus on:

```text
syntax
long-range dependencies
local patterns
semantic relations
```

But MQA forces them to use the **same K,V representation**.

---

### GQA Solution

By grouping heads:

```text
each group gets its own KV
```

So representation capacity increases.

Model quality becomes **much closer to MHA**.

---

# 🔟 Real Example

Model:

- Llama 3

Configuration example:

```text
num_heads = 32
num_kv_heads = 8
```

Grouping:

```text
4 query heads per KV head
```

Memory reduction:

```text
4× smaller KV cache
```

But quality remains strong.

---

# 1️⃣1️⃣ Complexity Comparison

| Attention | KV Memory | Quality  |
| --------- | --------- | -------- |
| MHA       | high      | best     |
| MQA       | very low  | worse    |
| GQA       | medium    | near MHA |

---

# 🎯 Key Insight

Grouped Query Attention is important because it **solves the trade-off between memory and model quality**.

It provides:

```text
near-MHA performance
+
MQA-level memory savings
```

That is why **modern LLMs widely use GQA**.

---

✅ **Summary**

Grouped Query Attention:

- groups multiple query heads to share KV
- reduces KV cache memory
- keeps attention diversity
- improves inference speed
- preserves model quality.

Used in modern models like:

- Llama 3
- Gemma
- Gemini

---
