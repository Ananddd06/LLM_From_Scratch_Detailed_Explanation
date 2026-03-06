# Example Configuration for a 120B Parameter LLM

| Parameter        | Value             | Explanation                 |
| ---------------- | ----------------- | --------------------------- |
| `vocab_size`     | **100000**        | tokenizer vocabulary        |
| `hidden_size`    | **12288**         | embedding / model dimension |
| `n_layers`       | **80**            | transformer blocks          |
| `n_head`         | **96**            | attention heads             |
| `n_kv_head`      | **12**            | KV heads (GQA)              |
| `head_dim`       | **128**           | per-head dimension          |
| `ffn_dim`        | **49152**         | feedforward dimension       |
| `context_length` | **8192**          | max tokens                  |
| `rope_theta`     | **10000**         | RoPE scaling                |
| `activation`     | **SiLU / SwiGLU** | activation function         |
| `norm_type`      | **RMSNorm**       | normalization               |
| `dropout`        | **0 or 0.1**      | regularization              |
| `tie_embeddings` | **True**          | share embedding weights     |

---

# Derived Values (Using Formulas)

### Head Dimension

```text
head_dim = hidden_size / n_head
```

```
12288 / 96 = 128
```

---

### Queries per KV Head (GQA)

```text
queries_per_kv = n_head / n_kv_head
```

```
96 / 12 = 8
```

So:

```
8 query heads share one KV head
```

---

### Feed Forward Dimension

Typical rule:

```text
ffn_dim ≈ 4 × hidden_size
```

```
4 × 12288 = 49152
```

---

# Tensor Shapes

If

```
B = batch
T = sequence length
```

Input tensor:

```
(B, T, 12288)
```

Attention tensors:

```
Q → (B, 96, T, 128)
K → (B, 12, T, 128)
V → (B, 12, T, 128)
```

---

# Embedding Parameters

Embedding matrix:

```
vocab_size × hidden_size
```

```
100000 × 12288 ≈ 1.23B parameters
```

---

# Approximate Total Parameters

A rough transformer estimate:

```
parameters ≈ 12 × layers × hidden_size²
```

Substitute:

```
12 × 80 × (12288²)
≈ 120B parameters
```

---

# Final Architecture Summary

```
vocab_size      = 100000
hidden_size     = 12288
n_layers        = 80
n_head          = 96
n_kv_head       = 12
head_dim        = 128
ffn_dim         = 49152
context_length  = 8192
rope_theta      = 10000
activation      = SiLU / SwiGLU
norm_type       = RMSNorm
dropout         = 0
tie_embeddings  = True
```

---

💡 **Key insight**

Large models scale mainly by increasing:

```
hidden_size
number_of_layers
attention_heads
```

Parameter growth roughly follows:

```
parameters ∝ layers × hidden_size²
```

---
