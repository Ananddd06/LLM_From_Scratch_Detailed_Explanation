### 🧠 Post-Norm in Transformers (Original Transformer Architecture)

Post-Norm is the **original normalization design used in the Transformer architecture (2017 – "Attention Is All You Need")**.

In Post-Norm, normalization is applied **after the sublayer and residual addition**, not before it.

This means the computation order becomes:

````

Attention → Residual Add → Norm
MLP → Residual Add → Norm

```id="postnorm_flow"

---

### 🏗 Basic Idea of Post-Norm

A transformer block contains two major components:

1️⃣ Self-Attention
2️⃣ Feedforward Network (MLP)

In **Post-Norm**, normalization happens **after these operations**.

````

x
↓
Self Attention
↓
Residual Add
↓
Normalization
↓
MLP
↓
Residual Add
↓
Normalization

```id="postnorm_structure"

So normalization stabilizes the **output of the residual connection**.

---

### 🧮 Mathematical Formulation

Let:

- \(x\) = input hidden state
- \(F\) = transformer sublayer (attention or MLP)

The Post-Norm transformer block is defined as:

$$
x_1 = \text{Norm}(x + \text{Attention}(x))
$$

$$
x_2 = \text{Norm}(x_1 + \text{MLP}(x_1))
$$

Final output:

$$
x_{out} = x_2
$$

---

### ⚙️ Attention Computation in Post-Norm

Inside the attention module, the hidden state is projected into query, key, and value vectors.

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

Where \(X = x\).

Then attention is computed as:

$$
\text{Attention}(Q,K,V) =
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This generates contextualized token representations.

---

### 🧩 Transformer Block Structure (Post-Norm)

A Post-Norm transformer block looks like this:

```

Input hidden state (x)
↓
Self Attention
↓
Residual Addition
↓
Normalization
↓
Feed Forward Network
↓
Residual Addition
↓
Normalization
↓
Output hidden state

````id="postnorm_block"

---

### 🧑‍💻 Example Code (Post-Norm Transformer Block)

```python
class Block(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = Attention(dim)
        self.mlp = MLP(dim)

    def forward(self, x):

        x = self.norm1(x + self.attn(x))

        x = self.norm2(x + self.mlp(x))

        return x
````

This code exactly follows the Post-Norm equations.

---

### 🔬 Why Post-Norm Was Used Initially

Post-Norm stabilizes the **output of each transformer layer**.

Advantages include:

✅ consistent output scale
✅ normalized residual outputs
✅ improved early training behavior

When transformers were shallow (6–12 layers), Post-Norm worked very well.

---

### 📉 Problem With Post-Norm in Deep Models

As transformer models became deeper (24+ layers), Post-Norm caused gradient instability.

The issue occurs during backpropagation.

For Post-Norm:

$$
x_{l+1} = \text{Norm}(x_l + F(x_l))
$$

Gradient becomes:

$$
\frac{\partial L}{\partial x_l}
===============================

\frac{\partial L}{\partial x_{l+1}}
\frac{\partial \text{Norm}}{\partial x_l}
\left(
1 + \frac{\partial F}{\partial x_l}
\right)
$$

Because gradients must pass through **normalization at every layer**, deep models may suffer from:

❌ unstable gradients
❌ slower convergence
❌ training divergence

---

### 📊 Post-Norm vs Pre-Norm

| Feature                | Post-Norm             | Pre-Norm        |
| ---------------------- | --------------------- | --------------- |
| Normalization position | After residual        | Before sublayer |
| Original transformer   | Yes                   | No              |
| Gradient stability     | Lower for deep models | High            |
| Used in modern LLMs    | Rare                  | Very common     |

---

### 🧠 Models That Used Post-Norm

Some important models used Post-Norm:

- Original Transformer (2017)
- BERT
- RoBERTa
- Early T5 variants

These models usually had **fewer transformer layers**, so Post-Norm was stable enough.

---

### 🚀 Evolution Toward Pre-Norm

Transformer normalization evolved in three stages:

````
Post-Norm + LayerNorm  (2017 Transformer)
            ↓
Pre-Norm + LayerNorm   (GPT-2, GPT-3)
            ↓
Pre-Norm + RMSNorm     (LLaMA, Mistral, DeepSeek)
``` id="norm_evolution"

As models became larger and deeper, Pre-Norm became the preferred architecture.

---

### 🎯 Key Takeaways

- Post-Norm applies normalization **after residual connections**.
- It was the **original design in the Transformer paper**.
- Works well for **shallow models**.
- Can cause **gradient instability in very deep transformers**.
- Modern LLMs mostly replaced it with **Pre-Norm architectures**.

Post-Norm played an important role in the **early development of transformers**, but modern large-scale models rely on **Pre-Norm for better training stability and scalability**.

````
