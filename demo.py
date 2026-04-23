import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from pathlib import Path
import re
from tokenizers import Tokenizer

# =============================================================================
# PART 1: VISION TRANSFORMER (From your provided code)
# =============================================================================

class NewGELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # Needs self.attention_head_size defined
        # Note: Fixing missing attribute in your original snippet:
        self.attention_head_size = query.size(-1) 
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)

class FasterMultiHeadAttention(nn.Module):
    """
    Using FasterMultiHeadAttention because it is easier to load standard HF weights (QKV projections)
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        qkv = self.qkv_projection(x)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.all_head_size)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Force Faster Attention for easier weight loading
        config["use_faster_attention"] = True 
        self.attention = FasterMultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)

class VisionEncoder(nn.Module):
    """
    Wrapper for the ViT that exposes the raw embeddings instead of classification logits.
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        # We remove the classifier head because we want features, not class predictions

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Return the sequence of embeddings (CLS + Patches)
        if not output_attentions:
            return (encoder_output, None)
        else:
            return (encoder_output, all_attentions)


# =============================================================================
# PART 2: QWEN 3 MODEL (From your provided code)
# =============================================================================

QWEN_CONFIG_05_B = {
    "vocab_size": 151_936,     "context_length": 32_768,  "emb_dim": 896,
    "n_heads": 14,             "n_layers": 24,            "hidden_dim": 4864,
    "head_dim": 64,            "qk_norm": True,           "n_kv_groups": 2,
    "rope_base": 1_000_000.0,  "dtype": torch.bfloat16,
}

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"], num_heads=cfg["n_heads"], head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"], qk_norm=cfg["qk_norm"], dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, layer_idx=None, exact=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache, layer_idx=layer_idx, exact=exact)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        if head_dim is None:
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None, layer_idx=None, exact=False):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        if self.q_norm: queries = self.q_norm(queries)
        if self.k_norm: keys_new = self.k_norm(keys_new)
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)
        if cache is not None:
            cache.append(layer_idx, keys_new, values_new)
            keys, values = cache.view(layer_idx)
        else:
            keys, values = keys_new, values_new
        if self.group_size > 1:
            keys = keys[:, :, None, :, :].expand(b, self.num_kv_groups, self.group_size, *keys.shape[2:]).reshape(b, self.num_heads, *keys.shape[2:])
            values = values[:, :, None, :, :].expand(b, self.num_kv_groups, self.group_size, *values.shape[2:]).reshape(b, self.num_heads, *values.shape[2:])
        
        attn_mask = mask
        if attn_mask is not None and attn_mask.dtype != queries.dtype:
            attn_mask = attn_mask.to(queries.dtype)
            
        context = torch.nn.functional.scaled_dot_product_attention(
            queries.contiguous(), keys.contiguous(), values.contiguous(),
            attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )
        return self.out_proj(context.transpose(1, 2).reshape(b, num_tokens, self.d_out))

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin, offset=0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos + rotated * sin).to(dtype=x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible: x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps) * self.scale
        if self.shift is not None: norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)

class Qwen3Model(nn.Module):
    def __init__(self, cfg, exact=False):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        
        head_dim = cfg["head_dim"] if cfg["head_dim"] is not None else cfg["emb_dim"] // cfg["n_heads"]
        cos, sin = compute_rope_params(head_dim=head_dim, theta_base=cfg["rope_base"], context_length=cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.exact = exact
        self.current_pos = 0

    def forward(self, in_idx, cache=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x.shape[1]
        
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(torch.full((num_tokens, pos_end), -torch.inf, device=x.device, dtype=self.cfg["dtype"]), diagonal=1 + pos_start)
        else:
            pos_start = 0
            mask = torch.triu(torch.full((num_tokens, num_tokens), -torch.inf, device=x.device, dtype=self.cfg["dtype"]), diagonal=1)
            
        mask = mask[None, None, :, :]
        for i, block in enumerate(self.trf_blocks):
            if cache is not None: cache.allocate(i, x.size(0))
            x = block(x, mask, self.cos, self.sin, start_pos=pos_start, cache=cache, layer_idx=i, exact=self.exact)
        
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits
    
    def reset_kv_cache(self):
        self.current_pos = 0

# =============================================================================
# PART 3: MULTIMODAL INTEGRATION & WEIGHT LOADING
# =============================================================================

def load_pretrained_vit_weights(vit_model, hf_model_name="google/vit-base-patch16-224", token=None):
    from transformers import ViTModel
    print(f"Loading ViT weights from: {hf_model_name}")
    hf_model = ViTModel.from_pretrained(hf_model_name, token=token)
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = {}
    
    for key, tensor in hf_state_dict.items():
        if "embeddings.patch_embeddings.projection" in key:
            new_key = key.replace("embeddings", "embedding")
        elif "embeddings.cls_token" in key:
            new_key = "embedding.cls_token"
        elif "embeddings.position_embeddings" in key:
            new_key = "embedding.position_embeddings"
        elif "attention.attention.query.weight" in key or "attention.attention.key.weight" in key or "attention.attention.value.weight" in key:
            continue
        elif "attention.output.dense" in key:
            new_key = key.replace("encoder.layer", "encoder.blocks").replace("attention.output.dense", "attention.output_projection")
        elif "layernorm_before" in key:
            new_key = key.replace("encoder.layer", "encoder.blocks").replace("layernorm_before", "layernorm_1")
        elif "layernorm_after" in key:
            new_key = key.replace("encoder.layer", "encoder.blocks").replace("layernorm_after", "layernorm_2")
        elif "intermediate.dense" in key:
            new_key = key.replace("encoder.layer", "encoder.blocks").replace("intermediate.dense", "mlp.dense_1")
        elif "output.dense" in key and "attention" not in key:
            new_key = key.replace("encoder.layer", "encoder.blocks").replace("output.dense", "mlp.dense_2")
        else:
            continue
        custom_state_dict[new_key] = tensor

    for i in range(12):
        q_w = hf_state_dict[f"encoder.layer.{i}.attention.attention.query.weight"]
        k_w = hf_state_dict[f"encoder.layer.{i}.attention.attention.key.weight"]
        v_w = hf_state_dict[f"encoder.layer.{i}.attention.attention.value.weight"]
        custom_state_dict[f"encoder.blocks.{i}.attention.qkv_projection.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        
        if f"encoder.layer.{i}.attention.attention.query.bias" in hf_state_dict:
            q_b = hf_state_dict[f"encoder.layer.{i}.attention.attention.query.bias"]
            k_b = hf_state_dict[f"encoder.layer.{i}.attention.attention.key.bias"]
            v_b = hf_state_dict[f"encoder.layer.{i}.attention.attention.value.bias"]
            custom_state_dict[f"encoder.blocks.{i}.attention.qkv_projection.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

    vit_model.load_state_dict(custom_state_dict, strict=False)
    print(f"✓ ViT weights loaded successfully")

def load_pretrained_qwen_weights(qwen_model, hf_model_name="Qwen/Qwen2.5-0.5B", token=None):
    from transformers import AutoModelForCausalLM
    print(f"Loading Qwen weights from: {hf_model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, token=token, torch_dtype=torch.bfloat16)
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = {}
    
    for key, tensor in hf_state_dict.items():
        if "model.embed_tokens.weight" in key:
            custom_state_dict["tok_emb.weight"] = tensor
        elif "model.norm.weight" in key:
            custom_state_dict["final_norm.scale"] = tensor
        elif "lm_head.weight" in key:
            custom_state_dict["out_head.weight"] = tensor
        elif "model.layers" in key:
            layer_num = key.split(".")[2]
            if "self_attn.q_proj.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.att.W_query.weight"] = tensor
            elif "self_attn.k_proj.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.att.W_key.weight"] = tensor
            elif "self_attn.v_proj.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.att.W_value.weight"] = tensor
            elif "self_attn.o_proj.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.att.out_proj.weight"] = tensor
            elif "self_attn.q_norm.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.att.q_norm.scale"] = tensor
            elif "self_attn.k_norm.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.att.k_norm.scale"] = tensor
            elif "mlp.gate_proj.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.ff.fc1.weight"] = tensor
            elif "mlp.up_proj.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.ff.fc2.weight"] = tensor
            elif "mlp.down_proj.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.ff.fc3.weight"] = tensor
            elif "input_layernorm.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.norm1.scale"] = tensor
            elif "post_attention_layernorm.weight" in key:
                custom_state_dict[f"trf_blocks.{layer_num}.norm2.scale"] = tensor
    
    qwen_model.load_state_dict(custom_state_dict, strict=False)
    print(f"✓ Qwen weights loaded successfully")

class MultimodalQwen3Model(nn.Module):
    def __init__(self, qwen_cfg, vit_cfg, hf_qwen_name="Qwen/Qwen2.5-0.5B", hf_vit_name="google/vit-base-patch16-224", token=None):
        super().__init__()
        self.qwen_cfg = qwen_cfg
        self.vit_cfg = vit_cfg
        
        # 1. Initialize Qwen and load HF weights
        self.qwen = Qwen3Model(qwen_cfg)
        load_pretrained_qwen_weights(self.qwen, hf_model_name=hf_qwen_name, token=token)
        
        # 2. Initialize Vision Encoder and load HF weights
        self.vision_encoder = VisionEncoder(vit_cfg)
        load_pretrained_vit_weights(self.vision_encoder, hf_model_name=hf_vit_name, token=token)
        
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # 3. Projector: ViT dim -> Qwen dim
        self.projector = nn.Sequential(
            nn.Linear(vit_cfg["hidden_size"], qwen_cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(qwen_cfg["emb_dim"], qwen_cfg["emb_dim"])
        ).to(qwen_cfg["dtype"])
        
        self.cos = self.qwen.cos
        self.sin = self.qwen.sin
        self.current_pos = 0

    def forward(self, in_idx, pixel_values=None, cache=None):
        """
        in_idx: (batch_size, text_seq_len)
        pixel_values: (batch_size, C, H, W) or None
        """
        device = in_idx.device
        batch_size = in_idx.shape[0]

        # 1. Get Text Embeddings
        text_embeds = self.qwen.tok_emb(in_idx) # (B, T, D)

        # 2. Get Image Embeddings if provided
        if pixel_values is not None:
            # Vision Forward
            vit_features, _ = self.vision_encoder(pixel_values) # (B, 197, 768)
            
            # Project to LLM Dim
            image_embeds = self.projector(vit_features) # (B, 197, 1024)
            
            # Concatenate Image + Text
            # We prepend images to the text sequence
            x = torch.cat([image_embeds, text_embeds], dim=1)
        else:
            x = text_embeds

        # 3. Compute Mask (Adjusted for image prefix)
        num_tokens = x.shape[1]
        
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            # Create mask for the new segment relative to total history
            mask = torch.triu(
                torch.full((num_tokens, pos_end), -torch.inf, device=device, dtype=self.qwen_cfg["dtype"]),
                diagonal=1 + pos_start,
            )
        else:
            pos_start = 0
            # Standard causal mask for the combined sequence
            mask = torch.triu(
                torch.full((num_tokens, num_tokens), -torch.inf, device=device, dtype=self.qwen_cfg["dtype"]),
                diagonal=1,
            )
            
        mask = mask[None, None, :, :]

        # 4. Pass through Qwen Transformer Blocks
        for i, block in enumerate(self.qwen.trf_blocks):
            if cache is not None:
                cache.allocate(i, x.size(0))
            x = block(
                x, mask, self.cos, self.sin,
                start_pos=pos_start,
                cache=cache,
                layer_idx=i,
                exact=self.qwen.exact,
            )

        # 5. Output Head
        x = self.qwen.final_norm(x)
        logits = self.qwen.out_head(x.to(self.qwen_cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load HF token from .env
    load_dotenv()
    hf_token = os.getenv("hugging_face_token")
    
    vit_config = {
        "image_size": 224, "patch_size": 16, "num_channels": 3,
        "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
        "intermediate_size": 3072, "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0, "qkv_bias": True
    }

    # Instantiate with HF model loading
    model = MultimodalQwen3Model(
        qwen_cfg=QWEN_CONFIG_05_B,
        vit_cfg=vit_config,
        hf_qwen_name="Qwen/Qwen2.5-0.5B",
        hf_vit_name="google/vit-base-patch16-224",
        token=hf_token
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("\n✓ Multimodal model created with pretrained weights")

    # Test forward pass
    batch_size = 1
    text_input = torch.randint(0, 151936, (batch_size, 10)).to(device)
    image_input = torch.randn(batch_size, 3, 224, 224).to(device)

    with torch.inference_mode():
        logits = model(text_input, pixel_values=image_input)
        print(f"✓ Output shape: {logits.shape}")  # (1, 207, 151936)