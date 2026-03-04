# 1️⃣ Scaled Dot-Product Attention (Base Attention)

<div align="center">
  <img src="https://substackcdn.com/image/fetch/%24s_%21jPid%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9fb987e9-1774-4f98-8b36-bc71ae4dfb0b_1400x638.png" width="500"/>
  <br><br>
  <img src="https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AMgMP9-ewpcZsgSvPgcHgxg.png" width="500"/>
</div>
<br>

This is the **core attention used in the original Transformer (2017)**.

### 💡 Idea

Each token computes:

$$
Attention(Q,K,V) = softmax(QK^T/\sqrt{d_k})V
$$

**Where:**

- **Q** = Query
- **K** = Key
- **V** = Value

The score (QK^T) tells **how much one token should attend to another token**. ([MachineLearningMastery.com][2])

### 🎯 Used in

- GPT
- BERT
- T5
- Almost all transformers

---

# 2️⃣ Self-Attention

<div align="center">
  <img src="https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AZPePnPodMZeehez9YFmr9A.png" width="500"/>
  <br><br>
  <img src="https://ar5iv.labs.arxiv.org/html/1904.02679/assets/images/head_view_1_combined.png" width="500"/>
  <br><br>
  <img src="https://sebastianraschka.com/images/blog/2023/self-attention-from-scratch/summary.png" width="500"/>
  <br><br>
  <img src="https://sebastianraschka.com/images/blog/2023/self-attention-from-scratch/transformer.png" width="500"/>
</div>
<br>

Here **every token attends to every other token in the same sequence**.

### 📊 Example sentence

```
"The cat sat on the mat"
```

Token **cat** attends to **sat**, **mat**, etc.

This helps the model understand **relationships between words**. ([datacamp.com][3])

---

# 3️⃣ Multi-Head Attention (MHA)

<div align="center">
  <img src="https://uvadlc-notebooks.readthedocs.io/en/latest/_images/transformer_architecture.svg" width="400"/>
  <br><br>
  <img src="https://miro.medium.com/v2/resize%3Afit%3A1400/1%2ADKNIOlVfbh9K1EqU5iDJKA.png" width="500"/>
  <br><br>
  <img src="https://d2l.ai/_images/multi-head-attention.svg" width="500"/>
</div>
<br>

Instead of **one attention**, the model uses **multiple attention heads**.

### 📊 Example

| Head   | Learns                |
| ------ | --------------------- |
| Head 1 | Grammar               |
| Head 2 | Semantic meaning      |
| Head 3 | Long-range dependency |
| Head 4 | Positional relation   |

Each head processes a **smaller embedding dimension**, and the outputs are **concatenated together**. ([Wikipedia][1])

---

# 4️⃣ Cross Attention

<div align="center">
  <img src="https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AxzvpKDgLm2A-D9C04V4rOw.png" width="500"/>
  <br><br>
  <img src="https://media.licdn.com/dms/image/v2/C5612AQFCmlFpnydcYg/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1589551792750?e=2147483647&t=Sr0U7wQbybBrsT5da_nxvkbgbS0JO0bYFrwmtITmIvI&v=beta" width="500"/>
  <br><br>
  <img src="https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AVUz1IjbEAcdW0ldtNHFsUA.png" width="500"/>
  <br><br>
  <img src="https://substackcdn.com/image/fetch/%24s_%21RG5-%21%2Cw_1200%2Ch_675%2Cc_fill%2Cf_jpg%2Cq_auto%3Agood%2Cfl_progressive%3Asteep%2Cg_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1610309f-20df-4534-acc9-7b6806be7cf5_1016x832.png" width="500"/>
</div>
<br>

Used when **two sequences interact**.

### 📊 Example

**Input:**

```
English → French translation
```

Decoder tokens attend to **encoder tokens**.

**So:**

```
Query → decoder
Key/Value → encoder
```

### 🎯 Used in

- T5
- BART
- Diffusion models
- Multimodal models. ([Medium][4])

---

# 5️⃣ Multi-Query Attention (MQA)

✨ **Modern LLM optimization**

### 💡 Idea

**Normal MHA:**

```
Each head has separate K and V
```

**MQA:**

```
Multiple Q heads
Single shared K and V
```

### ⚡ Benefits

- Smaller **KV cache**
- Faster inference

### 🎯 Used in

- PaLM
- Some GPT variants. ([Wikipedia][1])

---

# 6️⃣ Grouped Query Attention (GQA)

✨ **Improvement over MQA**

Instead of **1 KV for all heads**, we use **groups**.

### 📊 Example

```
16 heads
4 KV groups
```

### ⚡ Benefits

- Better quality than MQA
- Faster than MHA

### 🎯 Used in

- **LLaMA-2 / LLaMA-3**
- **Mistral**
- **Gemma**. ([IBM][5])

---

# 7️⃣ Sliding Window Attention

<div align="center">
  <img src="https://klu.ai/_next/static/media/klu-sliding-window-attention.4bed727d.png" width="500"/>
  <br><br>
  <img src="https://mlr.cdn-apple.com/media/Fig1_global_local_attention_c866eec574.png" width="500"/>
  <br><br>
  <img src="https://media.licdn.com/dms/image/v2/D5612AQF7_xOtM7surQ/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1721187600392?e=2147483647&t=-5YcxgthRd4M03OBvv3upOZMQnbib-Ob4TQee2ZXKvU&v=beta" width="500"/>
</div>
<br>

Instead of attending to **all tokens**, each token attends only to a **local window**.

### 📊 Example

```
Token i → attends to tokens [i-256 ... i+256]
```

### ⚡ Benefits

- Reduces complexity
- Supports **long context**

### 🎯 Used in

- **Mistral**
- **Longformer**
- **Gemma**. ([GeeksforGeeks][6])

---

# 8️⃣ Sparse Attention

### 💡 Idea

Instead of **N² attention**, only attend to **selected tokens**.

### 📊 Examples

- BigBird
- Longformer
- DeepSeek sparse attention

### ⚡ Benefits

- Handles **very long sequences**

---

# 9️⃣ Flash Attention

⚡ **Not a new architecture but a faster implementation**

### 💡 Idea

- Compute attention **in GPU memory blocks**
- Reduce memory movement

### ⚡ Benefits

- 2-3× faster training
- Supports long context. ([Wikipedia][1])

### 🎯 Used in

Almost all modern LLMs

---

# 🔟 Multi-Head Latent Attention (MLA)

🎯 **Used in DeepSeek models**

### 💡 Idea

Instead of storing large KV cache:

```
project hidden states → small latent KV
```

### ⚡ Benefits

- **Much smaller KV cache**
- Faster inference. ([Wikipedia][1])

---

# 🔥 New Research Attention Mechanisms

Recent papers propose even more types:

| Attention              | Used for             |
| ---------------------- | -------------------- |
| Linear Attention       | O(N) complexity      |
| Ring Attention         | Distributed training |
| Paged Attention        | Inference serving    |
| Neighborhood Attention | Vision transformers  |
| HiLo Attention         | Vision efficiency    |

---

# 📊 Summary (Modern LLM Attention)

| Attention                   | Used in               |
| --------------------------- | --------------------- |
| Scaled Dot Product          | Base transformer      |
| Self Attention              | All LLMs              |
| Multi Head Attention        | GPT / BERT            |
| Multi Query Attention       | PaLM                  |
| Grouped Query Attention     | LLaMA / Mistral       |
| Sliding Window Attention    | Mistral               |
| Sparse Attention            | Longformer / DeepSeek |
| Flash Attention             | Efficient GPU compute |
| Multi-Head Latent Attention | DeepSeek              |

---

✅ **Important Insight**

Modern LLMs mainly combine:

```
GQA + FlashAttention + RoPE + KV cache
```

### 🏗️ Example Architectures

- **LLaMA-3** → GQA + FlashAttention
- **Mistral** → GQA + Sliding Window
- **DeepSeek** → MLA + Sparse Attention

---

💡 **Want to Learn More?**

Since you are studying **LLM architecture**, I can also show you the **latest attention innovations (2024–2025)** like:

- **Paged Attention (vLLM)**
- **Ring Attention**
- **Linear Attention (Kimi Linear / RWKV style)**
- **Hybrid attention (Mamba + Transformer)**
