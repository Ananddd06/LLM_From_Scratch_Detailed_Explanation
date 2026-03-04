# 1️⃣ Scaled Dot-Product Attention (Base Attention)

![Image](https://www.researchgate.net/publication/361332728/figure/fig1/AS%3A1168225815531523%401655538147415/Attention-model-in-Transformer-a-Scaled-dot-product-attention-model-b-Multi-head.png)

![Image](https://www.researchgate.net/publication/391696986/figure/fig4/AS%3A11431281435777087%401747128418823/Diagram-of-the-query-key-value-and-self-attention-mechanism-The-input-vectors-x-is.ppm)

![Image](https://substackcdn.com/image/fetch/%24s_%21jPid%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9fb987e9-1774-4f98-8b36-bc71ae4dfb0b_1400x638.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AMgMP9-ewpcZsgSvPgcHgxg.png)

This is the **core attention used in the original Transformer (2017)**.

### Idea

Each token computes:

$$
Attention(Q,K,V) = softmax(QK^T/\sqrt{d_k})V
$$

Where

- **Q** = Query
- **K** = Key
- **V** = Value

The score (QK^T) tells **how much one token should attend to another token**. ([MachineLearningMastery.com][2])

### Used in

- GPT
- BERT
- T5
- Almost all transformers

---

# 2️⃣ Self-Attention

![Image](https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AZPePnPodMZeehez9YFmr9A.png)

![Image](https://ar5iv.labs.arxiv.org/html/1904.02679/assets/images/head_view_1_combined.png)

![Image](https://sebastianraschka.com/images/blog/2023/self-attention-from-scratch/summary.png)

![Image](https://sebastianraschka.com/images/blog/2023/self-attention-from-scratch/transformer.png)

Here **every token attends to every other token in the same sequence**.

Example sentence:

```
"The cat sat on the mat"
```

Token **cat** attends to **sat**, **mat**, etc.

This helps the model understand **relationships between words**. ([datacamp.com][3])

---

# 3️⃣ Multi-Head Attention (MHA)

![Image](https://uvadlc-notebooks.readthedocs.io/en/latest/_images/transformer_architecture.svg)

![Image](https://www.researchgate.net/publication/349787630/figure/fig4/AS%3A997835319283722%401614913887919/shows-how-multi-head-attention-works-Typically-the-number-of-heads-being-used-is-set-to.ppm)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2ADKNIOlVfbh9K1EqU5iDJKA.png)

![Image](https://d2l.ai/_images/multi-head-attention.svg)

Instead of **one attention**, the model uses **multiple attention heads**.

Example:

| Head   | Learns                |
| ------ | --------------------- |
| Head 1 | grammar               |
| Head 2 | semantic meaning      |
| Head 3 | long-range dependency |
| Head 4 | positional relation   |

Each head processes a **smaller embedding dimension**, and the outputs are **concatenated together**. ([Wikipedia][1])

---

# 4️⃣ Cross Attention

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AxzvpKDgLm2A-D9C04V4rOw.png)

![Image](https://media.licdn.com/dms/image/v2/C5612AQFCmlFpnydcYg/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1589551792750?e=2147483647&t=Sr0U7wQbybBrsT5da_nxvkbgbS0JO0bYFrwmtITmIvI&v=beta)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AVUz1IjbEAcdW0ldtNHFsUA.png)

![Image](https://substackcdn.com/image/fetch/%24s_%21RG5-%21%2Cw_1200%2Ch_675%2Cc_fill%2Cf_jpg%2Cq_auto%3Agood%2Cfl_progressive%3Asteep%2Cg_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1610309f-20df-4534-acc9-7b6806be7cf5_1016x832.png)

Used when **two sequences interact**.

Example:

Input:

```
English → French translation
```

Decoder tokens attend to **encoder tokens**.

So:

```
Query → decoder
Key/Value → encoder
```

Used in:

- T5
- BART
- diffusion models
- multimodal models. ([Medium][4])

---

# 5️⃣ Multi-Query Attention (MQA)

Modern LLM optimization.

### Idea

Normal MHA:

```
Each head has separate K and V
```

MQA:

```
Multiple Q heads
Single shared K and V
```

Benefits:

- Smaller **KV cache**
- Faster inference

Used in:

- PaLM
- some GPT variants. ([Wikipedia][1])

---

# 6️⃣ Grouped Query Attention (GQA)

Improvement over MQA.

Instead of **1 KV for all heads**, we use **groups**.

Example:

```
16 heads
4 KV groups
```

Benefits:

- better quality than MQA
- faster than MHA

Used in:

- **LLaMA-2 / LLaMA-3**
- **Mistral**
- **Gemma**. ([IBM][5])

---

# 7️⃣ Sliding Window Attention

![Image](https://klu.ai/_next/static/media/klu-sliding-window-attention.4bed727d.png)

![Image](https://mlr.cdn-apple.com/media/Fig1_global_local_attention_c866eec574.png)

![Image](https://media.licdn.com/dms/image/v2/D5612AQF7_xOtM7surQ/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1721187600392?e=2147483647&t=-5YcxgthRd4M03OBvv3upOZMQnbib-Ob4TQee2ZXKvU&v=beta)

![Image](https://klu.ai/_next/image?q=100&url=%2F_next%2Fstatic%2Fmedia%2Fklu-sliding-window-attention.4bed727d.png&w=3840)

Instead of attending to **all tokens**, each token attends only to a **local window**.

Example:

```
Token i → attends to tokens [i-256 ... i+256]
```

Benefits:

- reduces complexity
- supports **long context**

Used in:

- **Mistral**
- **Longformer**
- **Gemma**. ([GeeksforGeeks][6])

---

# 8️⃣ Sparse Attention

Idea:

Instead of **N² attention**, only attend to **selected tokens**.

Examples:

- BigBird
- Longformer
- DeepSeek sparse attention

Benefits:

- handles **very long sequences**.

---

# 9️⃣ Flash Attention

This is not a new architecture but a **faster implementation**.

Idea:

- compute attention **in GPU memory blocks**
- reduce memory movement.

Benefits:

- 2-3× faster training
- supports long context. ([Wikipedia][1])

Used in almost all modern LLMs.

---

# 🔟 Multi-Head Latent Attention (MLA)

Used in **DeepSeek models**.

Idea:

Instead of storing large KV cache:

```
project hidden states → small latent KV
```

Benefits:

- **much smaller KV cache**
- faster inference. ([Wikipedia][1])

---

# 🔥 New Research Attention Mechanisms

Recent papers propose even more types:

| Attention              | Used for             |
| ---------------------- | -------------------- |
| Linear Attention       | O(N) complexity      |
| Ring Attention         | distributed training |
| Paged Attention        | inference serving    |
| Neighborhood Attention | vision transformers  |
| HiLo Attention         | vision efficiency    |

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

✅ **Important insight**

Modern LLMs mainly combine:

```
GQA + FlashAttention + RoPE + KV cache
```

Example architectures:

- **LLaMA-3 → GQA + FlashAttention**
- **Mistral → GQA + Sliding Window**
- **DeepSeek → MLA + Sparse Attention**

---

✅ Since you are studying **LLM architecture**, I can also show you the **latest attention innovations (2024–2025)** like:

- **Paged Attention (vLLM)**
- **Ring Attention**
- **Linear Attention (Kimi Linear / RWKV style)**
- **Hybrid attention (Mamba + Transformer)**
