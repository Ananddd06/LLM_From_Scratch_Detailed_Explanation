### 🧠 Introduction to Layer Normalization in Transformers

Layer Normalization (**LayerNorm**) is one of the most important components in Transformer architectures. It stabilizes the training of deep neural networks by normalizing the activations within each token's feature vector.

Transformers process sequences where each token is represented as a vector. During training, these vectors can become very large or very small, which can cause unstable gradients and slow learning. LayerNorm keeps these values in a stable range so the model can train efficiently.

LayerNorm was first introduced in the paper **"Layer Normalization" (Ba et al., 2016)** and later used in the **Transformer architecture (Vaswani et al., 2017)**.

---

### ⚠️ Why Transformers Need Normalization

Transformers contain many stacked layers, each performing attention and feed-forward operations. These operations continuously change the scale of the activations.

Without normalization:

- Activations can grow very large 📈
- Activations can shrink toward zero 📉
- Gradients become unstable
- Deep models fail to converge

Normalization solves these issues.

Benefits of LayerNorm:

- Stabilizes gradients
- Improves convergence speed
- Allows training of deep models
- Prevents exploding activations

---

### 🏗 Where LayerNorm Appears in Transformers

In a transformer block, normalization is typically used before attention and before the feedforward network.

Modern architectures use **Pre-Norm**:

```

Input
↓
LayerNorm
↓
Multi-Head Attention
↓
Residual Add
↓
LayerNorm
↓
Feed Forward Network
↓
Residual Add

```

Mathematically:

$$
x = x + Attention(LayerNorm(x))
$$

$$
x = x + MLP(LayerNorm(x))
$$

This structure improves training stability for deep models.

---

### 🔬 What LayerNorm Actually Does

LayerNorm normalizes the features of a vector for each token independently.

If a token representation is:

```

x = [x₁, x₂, x₃, ... x_d]

```

where \(d\) is the embedding dimension.

LayerNorm performs three main steps:

1️⃣ Compute the mean of the vector  
2️⃣ Compute the variance  
3️⃣ Normalize the values

---

### 🧮 Step 1 — Compute the Mean

The mean of the vector is:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i
$$

This represents the **average value of all features** in the token vector.

---

### 🧮 Step 2 — Compute the Variance

Variance measures how spread out the values are:

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

Large variance means the values vary a lot.

---

### 🧮 Step 3 — Normalize the Vector

Now each value is normalized:

$$
\hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Where:

- \( \epsilon \) is a very small constant to avoid division by zero.

This transformation makes the vector have:

- Mean ≈ 0
- Variance ≈ 1

---

### 🎯 Step 4 — Apply Learnable Parameters

LayerNorm then applies two learnable parameters:

- **γ (gamma)** → scaling
- **β (beta)** → shifting

Final output:

$$
y_i = \gamma \hat{x_i} + \beta
$$

This allows the model to adjust the normalized values if needed.

---

### 🧩 Full LayerNorm Equation

Combining all steps:

$$
y_i =
\gamma
\frac{x_i - \mu}
{\sqrt{\sigma^2 + \epsilon}}
+ \beta
$$

Where:

- \( \mu \) = mean
- \( \sigma^2 \) = variance
- \( \gamma \) = scale parameter
- \( \beta \) = bias parameter

---

### 📊 Why LayerNorm Works Well in Transformers

Transformers operate on token embeddings.

Example token embedding:

```

[0.21, -0.33, 1.02, 0.44, -0.11, ...]

```

LayerNorm transforms it into a stable representation:

```

[-0.6, 0.3, 1.2, -0.1, ...]

```

Benefits:

- prevents numerical instability
- keeps features balanced
- ensures attention scores remain stable

---

### 🔁 LayerNorm vs BatchNorm

Transformers do **not use BatchNorm**.

| Feature                  | BatchNorm    | LayerNorm       |
| ------------------------ | ------------ | --------------- |
| Normalization axis       | Across batch | Across features |
| Dependency on batch size | Yes          | No              |
| Works with sequences     | Poorly       | Very well       |
| Used in CNNs             | Yes          | Rare            |
| Used in Transformers     | No           | Yes             |

BatchNorm depends on batch statistics, which makes it unsuitable for sequence models.

LayerNorm works independently for each token.

---

### 🏗 LayerNorm Inside Transformer Block

Full transformer block using LayerNorm:

```

x
↓
LayerNorm
↓
Attention
↓
Residual Add
↓
LayerNorm
↓
Feedforward Network
↓
Residual Add

```

Mathematically:

$$
x_1 = x + Attention(LayerNorm(x))
$$

$$
x_2 = x_1 + MLP(LayerNorm(x_1))
$$

Output:

$$
x_{out} = x_2
$$

---

### 🧑‍💻 PyTorch Implementation of LayerNorm

Simple implementation:

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)

        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * x_norm + self.beta
```

---

### 🧪 Example Usage

```python
batch = 2
seq_len = 5
dim = 8

x = torch.randn(batch, seq_len, dim)

norm = LayerNorm(dim)

y = norm(x)

print(y.shape)
```

Output:

```
torch.Size([2, 5, 8])
```

---

### 🧩 Transformer Block Using LayerNorm

Example transformer block:

```python
class TransformerBlock(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):

        x = x + self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x)
        )[0]

        x = x + self.mlp(self.norm2(x))

        return x
```

---

### 📌 Where LayerNorm Is Applied in an LLM Pipeline

Full pipeline:

```
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding Layer
 ↓
Transformer Blocks
      ├─ LayerNorm
      ├─ Attention
      ├─ Residual Add
      ├─ LayerNorm
      ├─ Feed Forward
      └─ Residual Add
 ↓
Final LayerNorm
 ↓
Linear Projection
 ↓
Softmax
 ↓
Next Token Prediction
```

---

### 🚀 Why Modern LLMs Prefer RMSNorm Instead

LayerNorm works well but has extra computations:

- mean subtraction
- variance calculation
- bias parameter

RMSNorm removes mean subtraction.

Advantages:

- fewer operations
- faster training
- lower memory usage

That is why modern models like **LLaMA and Mistral** use RMSNorm instead.

---

### 🎯 Key Takeaways

- LayerNorm stabilizes transformer training
- It normalizes each token's feature vector
- It uses **mean and variance normalization**
- It includes learnable scale and bias parameters
- It appears before attention and feedforward layers
- It enables training of deep transformer models

LayerNorm was the **original normalization method used in the Transformer architecture**, and it laid the foundation for modern normalization methods like **RMSNorm**.
