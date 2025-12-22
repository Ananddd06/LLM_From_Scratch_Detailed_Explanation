# Large Language Models (LLMs) — From Scratch

A clear and structured introduction to **Large Language Models**, covering  
**what they are, why they exist, and how they work internally**.

---

## 1. What is a Language Model?

A **Language Model (LM)** assigns probabilities to sequences of tokens in a language.

Formally, for a sequence of tokens:

$$
(w_1, w_2, w_3, \dots, w_n)
$$

the language model estimates:

$$
P(w_1, w_2, w_3, \dots, w_n)
$$

### Intuition
A language model learns **patterns in language** so that it can predict the **next token** given previous tokens.

---

## 2. What is a Large Language Model (LLM)?

A **Large Language Model (LLM)** is:
- A deep neural network
- Trained on massive text corpora
- With millions to trillions of parameters
- Designed to model natural language at scale

LLMs are typically built using the **Transformer architecture**.

---

## 3. Why Do We Need LLMs?

### Limitations of Earlier Approaches

#### Rule-Based Systems
- Manually written rules
- Not scalable
- Fail on unseen inputs

#### Classical Machine Learning
- Heavy feature engineering
- Limited context handling
- Poor generalization across tasks

#### Recurrent Models (RNNs / LSTMs)
- Sequential computation
- Vanishing gradient problems
- Inefficient for long contexts

### What LLMs Solve
- Learn representations automatically
- Handle long-range dependencies
- Generalize across multiple tasks
- Scale efficiently with data and compute

---

## 4. Tokens: The Basic Unit of LLMs

LLMs do not process raw text directly.  
Text is converted into **tokens**.

Tokens may represent:
- Words
- Subwords
- Characters
- Symbols

Example:

Input Text: "Hello, world!"  
Tokens: ["Hello", ",", "world", "!"]


---

## 5. Language Modeling Objective

The goal of a language model is to estimate:

$$
P(w_1, w_2, \dots, w_n)
$$

Using the **chain rule of probability**:

$$
P(w_1, w_2, \dots, w_n)
=
\prod_{i=1}^{n} P(w_i \mid w_1, \dots, w_{i-1})
$$

Each token is predicted **conditioned on all previous tokens**.

---

## 6. From N-gram Models to Neural Models

### N-gram Approximation

$$
P(w_i \mid w_1, \dots, w_{i-1})
\approx
P(w_i \mid w_{i-(n-1)}, \dots, w_{i-1})
$$

#### Problems
- Fixed context window
- Sparse statistics
- No semantic understanding

---

## 7. Neural Language Models and Embeddings

Neural models map tokens to dense vectors called **embeddings**:

$$
\text{token} \rightarrow \mathbb{R}^d
$$

These embeddings capture:
- Semantic meaning
- Syntactic structure
- Contextual similarity

---

## 8. Why RNNs and LSTMs Were Not Enough

RNN hidden state update:

$$
h_t = f(W_h h_{t-1} + W_x x_t)
$$

### Limitations
- Vanishing and exploding gradients
- Difficulty modeling long contexts
- Sequential computation prevents parallelism

---

## 9. Transformer Architecture

The **Transformer** removes recurrence and relies entirely on **self-attention**.

Key advantages:
- Full parallelization
- Long-range dependency modeling
- Efficient scaling

---

## 10. Self-Attention Mechanism

Each token produces:
- Query (Q)
- Key (K)
- Value (V)

Scaled dot-product attention:

$$
\text{Attention}(Q, K, V)
=
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

### Intuition
Each token decides **which other tokens to attend to** and how strongly.

---

## 11. Transformer Block Structure

A Transformer block consists of:
1. Multi-Head Self-Attention
2. Feed-Forward Network
3. Residual Connections
4. Layer Normalization

Feed-forward layer:

$$
\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2
$$

---

## 12. Building an LLM

An LLM is created by:
- Stacking many Transformer blocks
- Using large embedding dimensions
- Training on massive datasets

---

## 13. Training Objective

LLMs are trained using **next-token prediction**.

Loss function (cross-entropy):

$$
\mathcal{L}
=
- \sum_{i=1}^{n}
\log P(w_i \mid w_1, \dots, w_{i-1})
$$

---

## 14. Pretraining and Fine-tuning

### Pretraining
- Large-scale unlabeled text
- Learns grammar, facts, and patterns

### Fine-tuning
- Task-specific or instruction data
- Improves usability and alignment

---

## 15. Do LLMs Really Reason?

LLMs do not reason symbolically.  
They learn **statistical patterns of reasoning** from data.

Reasoning-like behavior emerges due to:
- Scale
- Diverse training data
- Architectural inductive biases

---

## 16. Why LLMs Matter

LLMs form the foundation for:
- AI assistants
- Code generation tools
- Search and retrieval systems
- Autonomous agents
- Scientific and educational tools

---

## Final Summary

A Large Language Model is:
- A probabilistic model of language
- Built using Transformers
- Trained via next-token prediction
- Scaled with data, parameters, and compute

Understanding these fundamentals enables you to **build, modify, and research LLMs from first principles**.
