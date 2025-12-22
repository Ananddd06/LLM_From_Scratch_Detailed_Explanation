# Transformers — In-Depth Explanation from Scratch

This document provides a **deep and structured explanation of the Transformer architecture**,  
covering **why it was invented, how it works mathematically, and why it scales so well for LLMs**.

---

## 1. What is a Transformer?

A **Transformer** is a neural network architecture designed to process **sequential data**  
using **self-attention instead of recurrence or convolution**.

It was introduced to solve major limitations of:

- RNNs
- LSTMs
- CNN-based sequence models

---

## 2. Why Were Transformers Needed?

### Problems with RNNs / LSTMs

RNN hidden state update:

$$
h_t = f(W_h h_{t-1} + W_x x_t)
$$

Limitations:

- Sequential computation (slow)
- Vanishing / exploding gradients
- Poor long-range dependency modeling
- Hard to scale on GPUs

---

## 3. Core Idea of Transformers

> **Replace recurrence with attention**

Instead of processing tokens one by one,  
**every token can directly attend to every other token**.

This allows:

- Full parallelization
- Global context access
- Better gradient flow

---

## 4. High-Level Transformer Architecture

A Transformer consists of stacked **Transformer blocks**.

Each block contains:

1. Self-Attention
2. Feed-Forward Network
3. Residual Connections
4. Layer Normalization

---

## 5. Token Embeddings

Tokens are mapped to dense vectors:

$$
\text{token} \rightarrow \mathbb{R}^d
$$

Embedding matrix:

$$
E \in \mathbb{R}^{V \times d}
$$

Where:

- \(V\) = vocabulary size
- \(d\) = embedding dimension

---

## 6. Positional Encoding

Since Transformers have **no recurrence**, they need explicit position information.

### Sinusoidal Positional Encoding

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

Final input:

$$
x = \text{Embedding} + \text{Positional Encoding}
$$

---

## 7. Self-Attention Mechanism

Each token generates:

- Query (Q)
- Key (K)
- Value (V)

Linear projections:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

---

## 8. Scaled Dot-Product Attention

Attention formula:

$$
\text{Attention}(Q, K, V)
=
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

### Intuition

- Queries ask **what to focus on**
- Keys represent **what each token offers**
- Values carry **actual information**

---

## 9. Why Scaling by \( \sqrt{d_k} \)?

Without scaling:

- Dot products grow large
- Softmax saturates
- Gradients vanish

Scaling stabilizes training:

$$
\frac{QK^\top}{\sqrt{d_k}}
$$

---

## 10. Multi-Head Attention

Instead of one attention operation, we use multiple heads:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Concatenation:

$$
\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
$$

### Why Multi-Head?

- Different heads learn different relationships
- Syntax, semantics, position, structure

---

## 11. Feed-Forward Network (FFN)

Applied **independently to each token**:

$$
\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2
$$

Typically:

- \( \sigma \) = ReLU / GELU / SwiGLU
- Expansion factor = 4× embedding size

---

## 12. Residual Connections

Residual path:

$$
x_{out} = x + \text{Sublayer}(x)
$$

Benefits:

- Prevents vanishing gradients
- Enables deep networks
- Faster convergence

---

## 13. Layer Normalization

LayerNorm:

$$
\text{LN}(x) = \frac{x - \mu}{\sigma} \gamma + \beta
$$

Stabilizes:

- Activations
- Gradient flow

---

## 14. Transformer Block Summary

Each block:

1. Self-Attention
2. Add & Norm
3. Feed-Forward
4. Add & Norm

Repeated **N times**.

---

## 15. Encoder vs Decoder Transformers

### Encoder

- Bidirectional attention
- Used for understanding tasks

### Decoder

- Causal (masked) attention
- Used for generation

---

## 16. Causal (Masked) Self-Attention

Mask ensures no future token leakage:

$$
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top + M}{\sqrt{d_k}}
\right)V
$$

Where:

- \(M = -\infty\) for future positions

---

## 17. Why Transformers Scale So Well

Key reasons:

- Parallel computation
- Stable gradients
- Modular architecture
- Hardware-friendly operations

Scaling laws apply:

- More data
- More parameters
- More compute → better performance

---

## 18. Transformers in LLMs

Modern LLMs are:

- Decoder-only Transformers
- Trained with next-token prediction
- Optimized for massive scale

Training objective:

$$
\mathcal{L}
=
- \sum_{i=1}^{n}
\log P(w_i \mid w_1,\dots,w_{i-1})
$$

---

## 19. Strengths and Limitations

### Strengths

- Long-range dependencies
- Parallelism
- Emergent reasoning

### Limitations

- Quadratic attention cost
- Memory intensive
- Data-hungry

---

## 20. Final Intuition

> A Transformer is a **differentiable information routing system**  
> where each token dynamically decides **who to listen to**.

---

## Final Summary

- Transformers replace recurrence with attention
- Self-attention is the core operation
- Stacking blocks creates expressive models
- Transformers are the backbone of modern LLMs

Understanding Transformers deeply is **non-negotiable** for LLM research and engineering.
