# GPT Model Implementation - Detailed Explanation

## Overview

This notebook demonstrates a complete implementation of a GPT (Generative Pre-trained Transformer) model from scratch, followed by loading pre-trained weights from HuggingFace's GPT-2 model. The implementation covers all core components needed to build and run a language model.

---

## Table of Contents

1. [Architecture Components](#architecture-components)
2. [Data Processing](#data-processing)
3. [Model Configuration](#model-configuration)
4. [Weight Loading from Pre-trained Models](#weight-loading)
5. [Text Generation](#text-generation)
6. [Key Implementation Details](#key-implementation-details)

---

## Architecture Components

### 1. Multi-Head Attention (`MultiHeadAttention`)

The attention mechanism is the core of the Transformer architecture.

**Key Features:**
- Splits the embedding dimension across multiple attention heads
- Implements causal masking to prevent attending to future tokens
- Uses Query, Key, Value projections

**Mathematical Flow:**

```
Input: x with shape (batch_size, num_tokens, d_in)

1. Linear Projections:
   Q = x @ W_query  → (batch_size, num_tokens, d_out)
   K = x @ W_key    → (batch_size, num_tokens, d_out)
   V = x @ W_value  → (batch_size, num_tokens, d_out)

2. Reshape for Multi-Head:
   Q, K, V → (batch_size, num_heads, num_tokens, head_dim)
   where head_dim = d_out / num_heads

3. Scaled Dot-Product Attention:
   scores = (Q @ K^T) / sqrt(head_dim)
   scores = mask_future_tokens(scores)
   weights = softmax(scores)
   context = weights @ V

4. Combine Heads:
   context → (batch_size, num_tokens, d_out)
   output = context @ W_out
```

**Code Highlights:**
```python
# Head dimension calculation
self.head_dim = d_out // num_heads

# Causal mask prevents looking at future tokens
self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

# Attention computation with masking
attn_scores.masked_fill_(mask_bool, -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```

---

### 2. Layer Normalization (`LayerNorm`)

Normalizes activations across the feature dimension to stabilize training.

**Formula:**
```
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β

where:
- μ = mean(x) across feature dimension
- σ² = variance(x) across feature dimension
- γ (scale) and β (shift) are learnable parameters
- ε = small constant for numerical stability (1e-5)
```

**Why It's Important:**
- Reduces internal covariate shift
- Allows higher learning rates
- Makes training more stable

---

### 3. GELU Activation (`GELU`)

Gaussian Error Linear Unit - a smooth activation function used in GPT models.

**Formula:**
```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

**Advantages over ReLU:**
- Smooth and differentiable everywhere
- Non-monotonic (can have negative outputs)
- Better gradient flow

---

### 4. Feed-Forward Network (`FeedForward`)

A two-layer MLP that processes each token independently.

**Architecture:**
```
Input (emb_dim) 
  → Linear (emb_dim → 4*emb_dim)
  → GELU
  → Linear (4*emb_dim → emb_dim)
  → Output (emb_dim)
```

**Purpose:**
- Adds non-linearity and capacity to the model
- The 4x expansion allows the model to learn complex transformations
- Applied position-wise (same operation for each token)

---

### 5. Transformer Block (`TransformerBlock`)

Combines attention and feed-forward layers with residual connections and layer normalization.

**Architecture (Pre-Norm):**
```
x → LayerNorm → MultiHeadAttention → Dropout → (+) → x'
x' → LayerNorm → FeedForward → Dropout → (+) → output

where (+) represents residual/skip connections
```

**Key Design Choices:**
- **Pre-Norm**: LayerNorm applied before attention/FFN (more stable training)
- **Residual Connections**: Allow gradients to flow directly through the network
- **Dropout**: Applied after attention and FFN for regularization

---

### 6. Complete GPT Model (`GPTModel`)

The full model architecture combining all components.

**Components:**

1. **Token Embedding** (`tok_emb`):
   - Maps token IDs to dense vectors
   - Shape: (vocab_size, emb_dim)

2. **Position Embedding** (`pos_emb`):
   - Adds positional information to tokens
   - Shape: (context_length, emb_dim)
   - Learned embeddings (not sinusoidal)

3. **Transformer Blocks** (`trf_blocks`):
   - Stack of N identical transformer blocks
   - GPT-2 Small: 12 blocks

4. **Final Layer Norm** (`final_norm`):
   - Normalizes output before prediction

5. **Output Head** (`out_head`):
   - Projects to vocabulary size for next-token prediction
   - Shape: (emb_dim, vocab_size)
   - Weight tied with token embedding

**Forward Pass:**
```python
def forward(self, in_idx):
    # 1. Embed tokens and add positions
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len))
    x = tok_embeds + pos_embeds
    
    # 2. Apply dropout
    x = self.drop_emb(x)
    
    # 3. Pass through transformer blocks
    x = self.trf_blocks(x)
    
    # 4. Final normalization
    x = self.final_norm(x)
    
    # 5. Project to vocabulary
    logits = self.out_head(x)
    return logits
```

---

## Data Processing

### Dataset Class (`GPTDatasetV1`)

Creates training samples using a sliding window approach.

**How It Works:**

```python
Text: "The quick brown fox jumps"
Tokenized: [464, 2068, 7586, 21831, 18045]

With max_length=3, stride=2:

Sample 1:
  Input:  [464, 2068, 7586]
  Target: [2068, 7586, 21831]

Sample 2:
  Input:  [7586, 21831, 18045]
  Target: [21831, 18045, <next>]
```

**Parameters:**
- `max_length`: Maximum sequence length
- `stride`: Step size for sliding window (smaller = more overlap)

**Purpose:**
- Creates input-target pairs for next-token prediction
- Overlapping windows ensure all tokens are seen in different contexts

---

## Model Configuration

### GPT-2 Model Sizes

The notebook supports loading different GPT-2 variants:

| Model | Parameters | Layers | Heads | Embedding Dim |
|-------|-----------|--------|-------|---------------|
| GPT-2 Small | 124M | 12 | 12 | 768 |
| GPT-2 Medium | 355M | 24 | 16 | 1024 |
| GPT-2 Large | 774M | 36 | 20 | 1280 |
| GPT-2 XL | 1558M | 48 | 25 | 1600 |

**Configuration Dictionary:**
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # GPT-2 tokenizer vocabulary
    "context_length": 1024,   # Maximum sequence length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of transformer blocks
    "drop_rate": 0.1,         # Dropout probability
    "qkv_bias": False         # Bias in attention projections
}
```

---

## Weight Loading from Pre-trained Models

### Understanding Weight Mapping

The notebook loads pre-trained GPT-2 weights from HuggingFace. Key differences in weight organization:

**HuggingFace Format:**
- Combined QKV weights: `h.{layer}.attn.c_attn.weight` with shape (emb_dim, 3*emb_dim)
- Conv1D layers (transposed linear layers)

**Custom Implementation:**
- Separate Q, K, V weights
- Standard PyTorch Linear layers

### Weight Loading Process

**1. Attention Weights:**
```python
# HuggingFace stores Q, K, V concatenated
c_attn_weight = d[f"h.{b}.attn.c_attn.weight"]  # (768, 2304)

# Split into Q, K, V
q_w, k_w, v_w = np.split(c_attn_weight, 3, axis=-1)
# Each is now (768, 768)

# Transpose because HuggingFace uses Conv1D (transposed)
gpt.trf_blocks[b].att.W_query.weight = q_w.T
gpt.trf_blocks[b].att.W_key.weight = k_w.T
gpt.trf_blocks[b].att.W_value.weight = v_w.T
```

**2. Embedding Weights:**
```python
# Token embeddings
gpt.tok_emb.weight = d["wte.weight"]

# Position embeddings
gpt.pos_emb.weight = d["wpe.weight"]

# Output head (weight tying)
gpt.out_head.weight = d["wte.weight"]
```

**3. Feed-Forward Weights:**
```python
# First layer (expansion)
gpt.trf_blocks[b].ff.layers[0].weight = d[f"h.{b}.mlp.c_fc.weight"].T

# Second layer (projection)
gpt.trf_blocks[b].ff.layers[2].weight = d[f"h.{b}.mlp.c_proj.weight"].T
```

**4. Normalization Parameters:**
```python
# Layer norm in attention block
gpt.trf_blocks[b].norm1.scale = d[f"h.{b}.ln_1.weight"]
gpt.trf_blocks[b].norm1.shift = d[f"h.{b}.ln_1.bias"]

# Layer norm in FFN block
gpt.trf_blocks[b].norm2.scale = d[f"h.{b}.ln_2.weight"]
gpt.trf_blocks[b].norm2.shift = d[f"h.{b}.ln_2.bias"]

# Final layer norm
gpt.final_norm.scale = d["ln_f.weight"]
gpt.final_norm.shift = d["ln_f.bias"]
```

### Weight Shape Verification

The `assign_check` function ensures shape compatibility:
```python
def assign_check(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())
```

---

## Text Generation

### Simple Generation (`generate_text_simple`)

Basic greedy decoding - always picks the most likely next token.

**Algorithm:**
```
1. Start with input tokens
2. For each new token to generate:
   a. Get model predictions (logits)
   b. Take argmax to get most likely token
   c. Append to sequence
3. Return generated sequence
```

**Limitations:**
- Deterministic (same input → same output)
- Can be repetitive
- No diversity in outputs

---

### Advanced Generation (`generate`)

Implements temperature scaling and top-k sampling for better generation quality.

**Temperature Scaling:**
```python
logits = logits / temperature

# temperature < 1.0: More confident (sharper distribution)
# temperature = 1.0: No change
# temperature > 1.0: More random (flatter distribution)
```

**Effect of Temperature:**
```
Original logits: [2.0, 1.0, 0.5]

Temperature = 0.5 (confident):
  Scaled: [4.0, 2.0, 1.0]
  Probs: [0.84, 0.14, 0.02]  ← More peaked

Temperature = 2.0 (random):
  Scaled: [1.0, 0.5, 0.25]
  Probs: [0.50, 0.30, 0.20]  ← More uniform
```

**Top-K Sampling:**
```python
# Keep only top k most likely tokens
top_logits, _ = torch.topk(logits, k)
min_val = top_logits[:, -1]

# Set all other logits to -inf
logits = torch.where(
    logits < min_val, 
    torch.tensor(float("-inf")), 
    logits
)
```

**Why Top-K Helps:**
- Prevents sampling very unlikely tokens
- Maintains diversity while avoiding nonsense
- k=50 is a common choice

**Complete Generation Process:**
```
1. Get logits from model
2. Apply top-k filtering (optional)
3. Apply temperature scaling (optional)
4. Convert to probabilities (softmax)
5. Sample from distribution
6. Append to sequence
7. Check for end-of-sequence token
8. Repeat
```

---

## Key Implementation Details

### 1. Causal Masking

Prevents the model from "cheating" by looking at future tokens:

```python
# Create upper triangular matrix
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

# Example for seq_len=4:
# [[0, 1, 1, 1],
#  [0, 0, 1, 1],
#  [0, 0, 0, 1],
#  [0, 0, 0, 0]]

# Apply mask by setting future positions to -inf
attn_scores.masked_fill_(mask.bool(), -torch.inf)
```

After softmax, -inf becomes 0, effectively blocking attention to future tokens.

---

### 2. Residual Connections

Allow gradients to flow directly through the network:

```python
# Without residual:
x = attention(x)  # Gradient must flow through attention

# With residual:
x = x + attention(x)  # Gradient can bypass attention
```

**Benefits:**
- Prevents vanishing gradients in deep networks
- Allows training very deep models (GPT-3: 96 layers)
- Provides identity mapping as fallback

---

### 3. Pre-Norm vs Post-Norm

**Pre-Norm (used in this implementation):**
```python
x = x + attention(norm(x))
x = x + ffn(norm(x))
```

**Post-Norm (original Transformer):**
```python
x = norm(x + attention(x))
x = norm(x + ffn(x))
```

**Why Pre-Norm:**
- More stable training
- Can use higher learning rates
- Better for very deep models
- Used in GPT-2, GPT-3

---

### 4. Weight Tying

The output projection shares weights with token embedding:

```python
self.tok_emb = nn.Embedding(vocab_size, emb_dim)
self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

# After loading:
self.out_head.weight = self.tok_emb.weight
```

**Benefits:**
- Reduces parameters (50257 * 768 = ~38M parameters saved)
- Improves generalization
- Standard practice in language models

---

### 5. Tokenization

Uses TikToken (OpenAI's tokenizer) with BPE encoding:

```python
tokenizer = tiktoken.get_encoding("gpt2")

# Encode text to token IDs
token_ids = tokenizer.encode("Hello, world!")
# [15496, 11, 995, 0]

# Decode back to text
text = tokenizer.decode(token_ids)
# "Hello, world!"
```

**Vocabulary Size:** 50,257 tokens
- 50,000 BPE merges
- 256 byte tokens
- 1 end-of-text token

---

## Example Usage

### Loading and Generating Text

```python
# 1. Load pre-trained model
gpt_hf = GPT2Model.from_pretrained("openai-community/gpt2")

# 2. Create custom model
gpt = GPTModel(GPT_CONFIG_124M)

# 3. Load weights
load_weights(gpt, gpt_hf)

# 4. Generate text
tokenizer = tiktoken.get_encoding("gpt2")
input_ids = text_to_token_ids("Hello, I am", tokenizer)

output_ids = generate(
    model=gpt,
    idx=input_ids,
    max_new_tokens=50,
    context_size=1024,
    temperature=1.0,
    top_k=50
)

output_text = token_ids_to_text(output_ids, tokenizer)
print(output_text)
```

### Output Example

```
Input: "What are Herbivorous"

Output: "What are Herbivorous and how effective can eating produce them 
(and why have people avoided "Herbivores".) As to where do they grow 
on an island, you can either find the food your plant is best adapted 
for eating locally..."
```

---

## Performance Considerations

### Memory Usage

**Model Size (GPT-2 Small):**
- Parameters: 124M
- Memory (FP32): 124M * 4 bytes = ~496 MB
- Memory (FP16): 124M * 2 bytes = ~248 MB

**Activation Memory:**
- Depends on batch size and sequence length
- For batch_size=1, seq_len=1024: ~100 MB

### Inference Speed

**Factors Affecting Speed:**
1. **Sequence Length**: Attention is O(n²)
2. **Batch Size**: Larger batches = better GPU utilization
3. **Model Size**: More parameters = slower
4. **Device**: GPU >> CPU

**Optimization Techniques:**
- Use FP16/BF16 precision
- KV-cache for autoregressive generation
- Flash Attention for memory efficiency
- Batch multiple sequences together

---

## Common Issues and Solutions

### 1. Shape Mismatches

**Problem:** Weight shapes don't match when loading
**Solution:** Check transpose operations and dimension ordering

### 2. Out of Memory

**Problem:** GPU runs out of memory
**Solutions:**
- Reduce batch size
- Reduce sequence length
- Use gradient checkpointing
- Use smaller model

### 3. Poor Generation Quality

**Problem:** Generated text is repetitive or nonsensical
**Solutions:**
- Adjust temperature (try 0.7-1.5)
- Use top-k sampling (k=40-50)
- Add top-p (nucleus) sampling
- Check if model loaded correctly

### 4. Slow Generation

**Problem:** Text generation is very slow
**Solutions:**
- Move model to GPU
- Implement KV-cache
- Use batch generation
- Consider model quantization

---

## Summary

This implementation demonstrates:

1. **Complete GPT Architecture**: All components from scratch
2. **Pre-trained Weight Loading**: Compatible with HuggingFace models
3. **Text Generation**: Multiple sampling strategies
4. **Production-Ready Code**: Proper error checking and device handling

**Key Takeaways:**
- Transformers are built from simple components (attention, FFN, norm)
- Residual connections and layer normalization are crucial for training
- Pre-trained weights can be loaded with careful weight mapping
- Generation quality depends on sampling strategy (temperature, top-k)

**Next Steps:**
- Implement training loop for fine-tuning
- Add more advanced sampling (top-p, beam search)
- Optimize inference with KV-cache
- Experiment with different model sizes
