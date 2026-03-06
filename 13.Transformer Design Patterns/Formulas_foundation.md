# 1️⃣ Hidden Size (Embedding Dimension)

This is the **main dimension of the model**.

```
hidden_size = embedding_dimension
```

Example:

```
hidden_size = 4096
```

Every token is represented as:

```
token_vector ∈ R^4096
```

---

# 2️⃣ Head Dimension

Attention splits the hidden size across heads.

Formula:

```
head_dim = hidden_size / n_head
```

Example:

```
hidden_size = 4096
n_head = 32
```

```
head_dim = 4096 / 32 = 128
```

So each attention head works on **128 dimensions**.

---

# 3️⃣ Query Heads vs KV Heads (GQA)

Formula:

```
queries_per_kv = n_head / n_kv_head
```

Example:

```
n_head = 64
n_kv_head = 8
```

```
queries_per_kv = 64 / 8 = 8
```

Meaning **8 query heads share the same K and V**.

---

# 4️⃣ Embedding Matrix Size

The embedding layer converts tokens → vectors.

Formula:

```
embedding_matrix = vocab_size × hidden_size
```

Example:

```
vocab_size = 32000
hidden_size = 4096
```

```
32000 × 4096
```

Parameters ≈ **131M**.

---

# 5️⃣ Output Projection (LM Head)

The output layer predicts next tokens.

Formula:

```
output_matrix = hidden_size × vocab_size
```

Example:

```
4096 × 32000
```

Many models **tie weights** with embeddings:

```
embedding_matrix = output_matrix^T
```

---

# 6️⃣ Attention Projection Dimensions

Each attention layer computes Q, K, V.

```
Q = hidden_size × hidden_size
K = hidden_size × hidden_size
V = hidden_size × hidden_size
```

Example:

```
4096 × 4096
```

But with **GQA**:

```
K,V size = hidden_size × (n_kv_head × head_dim)
```

---

# 7️⃣ Attention Tensor Shapes

If

```
B = batch size
T = sequence length
```

Then:

Queries

```
(B, n_head, T, head_dim)
```

Keys

```
(B, n_kv_head, T, head_dim)
```

Values

```
(B, n_kv_head, T, head_dim)
```

---

# 8️⃣ Feed Forward Network (MLP)

Most transformers expand hidden size.

Typical formula:

```
ffn_dim ≈ 4 × hidden_size
```

Example:

```
hidden_size = 4096
```

```
ffn_dim = 16384
```

Layer structure:

```
4096 → 16384 → 4096
```

Some models use:

```
ffn_dim ≈ 3.5 × hidden_size
```

---

# 9️⃣ Total Attention Output Shape

After attention:

```
(B, n_head, T, head_dim)
```

Merge heads:

```
(B, T, hidden_size)
```

Because:

```
hidden_size = n_head × head_dim
```

---

# 🔟 Important Core Relationship

One of the most important formulas:

```
hidden_size = n_head × head_dim
```

Example:

```
4096 = 32 × 128
```

---

# Summary Table

| Component        | Formula                           |
| ---------------- | --------------------------------- |
| Head dimension   | `head_dim = hidden_size / n_head` |
| Hidden size      | `hidden_size = n_head × head_dim` |
| Queries per KV   | `n_head / n_kv_head`              |
| Embedding matrix | `vocab_size × hidden_size`        |
| Output matrix    | `hidden_size × vocab_size`        |
| FFN dimension    | `≈ 4 × hidden_size`               |

---

✅ **Minimal parameter set needed to define a transformer**

Usually these are enough:

```
vocab_size
hidden_size
n_layers
n_head
n_kv_head
context_length
```

# Core Parameters (Must Have)

These are the **most important parameters** that define the model.

| Parameter        | Meaning                       |
| ---------------- | ----------------------------- |
| `vocab_size`     | number of tokens in tokenizer |
| `hidden_size`    | embedding dimension           |
| `n_layers`       | number of transformer blocks  |
| `n_head`         | number of attention heads     |
| `n_kv_head`      | KV heads (for GQA/MQA)        |
| `context_length` | maximum sequence length       |

These define the **main architecture**.

---

# Parameters Derived From Them

Once the above are known, several values can be computed.

### Head dimension

```
head_dim = hidden_size / n_head
```

---

### Queries per KV (GQA)

```
queries_per_kv = n_head / n_kv_head
```

---

### Feedforward dimension

Usually:

```
ffn_dim ≈ 4 × hidden_size
```

Example:

```
4096 → 16384
```

---

### Embedding matrix

```
embedding = vocab_size × hidden_size
```

---

### Attention shapes

```
Q → (B, n_head, T, head_dim)
K → (B, n_kv_head, T, head_dim)
V → (B, n_kv_head, T, head_dim)
```

---

# Additional Hyperparameters (Usually Needed)

Real models like **LLaMA 2** also define a few extra settings.

| Parameter        | Purpose                           |
| ---------------- | --------------------------------- |
| `ffn_dim`        | feedforward hidden size           |
| `rope_theta`     | RoPE positional embedding scaling |
| `dropout`        | regularization                    |
| `norm_type`      | LayerNorm / RMSNorm               |
| `activation`     | GELU / SiLU                       |
| `tie_embeddings` | share input/output embeddings     |

These **control training behavior** but not the basic structure.

---

# Minimal Transformer Config

A minimal config usually looks like this:

```
vocab_size
hidden_size
n_layers
n_head
n_kv_head
context_length
ffn_dim
```

That’s enough to build a working LLM.

---

# Example (similar to LLaMA-2 7B)

```
vocab_size = 32000
hidden_size = 4096
n_layers = 32
n_head = 32
n_kv_head = 32
context_length = 4096
ffn_dim = 11008
```

From this, everything else can be derived.

---

✅ **Final answer**

Your list:

```
vocab_size
hidden_size
n_layers
n_head
n_kv_head
context_length
```

is **almost enough**, but usually we also include:

```
ffn_dim
```

to fully define the transformer block.

---
