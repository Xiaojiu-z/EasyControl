import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from diffusers.models.attention_processor import Attention
    
class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        cond_width=512,
        cond_height=512,
        number=0,
        n_loras=1
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        
        self.cond_height = cond_height
        self.cond_width = cond_width
        self.number = number
        self.n_loras = n_loras

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        ####
        batch_size = hidden_states.shape[0]
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        block_size =  hidden_states.shape[1] - cond_size * self.n_loras
        shape = (batch_size, hidden_states.shape[1], 3072)
        mask = torch.ones(shape, device=hidden_states.device, dtype=dtype) 
        mask[:, :block_size+self.number*cond_size, :] = 0
        mask[:, block_size+(self.number+1)*cond_size:, :] = 0
        hidden_states = mask * hidden_states
        ####
        
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
    

class MultiSingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, ranks=[], lora_weights=[], network_alphas=[], device=None, dtype=None, cond_width=512, cond_height=512, n_loras=1):
        super().__init__()
        # Initialize a list to store the LoRA layers
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height
        
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights
        self.bank_attn = None
        self.bank_kv = []
        

    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond = False
    ) -> torch.FloatTensor:

        batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        scaled_seq_len = hidden_states.shape[1]
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        block_size =  scaled_seq_len - cond_size * self.n_loras
        scaled_cond_size = cond_size
        scaled_block_size = block_size

        if len(self.bank_kv)== 0:
            cache = True
        else:
            cache = False
        
        if cache:
            query = attn.to_q(hidden_states) 
            key = attn.to_k(hidden_states) 
            value = attn.to_v(hidden_states) 
            for i in range(self.n_loras):
                query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
                key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
                value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            self.bank_kv.append(key[:, :, scaled_block_size:, :])
            self.bank_kv.append(value[:, :, scaled_block_size:, :])
            
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
        
            num_cond_blocks = self.n_loras
            mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
            mask[ :scaled_block_size, :] = 0  # First block_size row
            for i in range(num_cond_blocks):
                start = i * scaled_cond_size + scaled_block_size
                end = (i + 1) * scaled_cond_size + scaled_block_size
                mask[start:end, start:end] = 0  # Diagonal blocks
            mask = mask * -1e20
            mask = mask.to(query.dtype)

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=mask)
            self.bank_attn = hidden_states[:, :, scaled_block_size:, :]
            
        else:
            query = attn.to_q(hidden_states) 
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            inner_dim = query.shape[-1]
            head_dim = inner_dim // attn.heads
            
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            key = torch.concat([key[:, :, :scaled_block_size, :], self.bank_kv[0]], dim=-2)
            value = torch.concat([value[:, :, :scaled_block_size, :], self.bank_kv[1]], dim=-2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            
            query = query[:, :, :scaled_block_size, :]

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=None)
            hidden_states = torch.concat([hidden_states, self.bank_attn], dim=-2)
            
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        cond_hidden_states = hidden_states[:, block_size:,:]
        hidden_states = hidden_states[:, : block_size,:]

        return hidden_states if not use_cond else (hidden_states, cond_hidden_states)


class MultiDoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, ranks=[], lora_weights=[], network_alphas=[], device=None, dtype=None, cond_width=512, cond_height=512, n_loras=1):
        super().__init__()
        
        # Initialize a list to store the LoRA layers
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.proj_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights
        self.bank_attn = None
        self.bank_kv = []


    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond=False,
    ) -> torch.FloatTensor:
        
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64 
        block_size =  hidden_states.shape[1] - cond_size * self.n_loras
        scaled_seq_len = encoder_hidden_states.shape[1] + hidden_states.shape[1]
        scaled_cond_size = cond_size
        scaled_block_size = scaled_seq_len - scaled_cond_size * self.n_loras

        # `context` projections.
        inner_dim = 3072
        head_dim = inner_dim // attn.heads
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states) 
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
        
        if len(self.bank_kv)== 0:
            cache = True
        else:
            cache = False
        
        if cache:
            
            query = attn.to_q(hidden_states) 
            key = attn.to_k(hidden_states) 
            value = attn.to_v(hidden_states) 
            for i in range(self.n_loras):
                query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
                key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
                value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            
            self.bank_kv.append(key[:, :, block_size:, :])
            self.bank_kv.append(value[:, :, block_size:, :])

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            
            num_cond_blocks = self.n_loras
            mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
            mask[ :scaled_block_size, :] = 0  # First block_size row
            for i in range(num_cond_blocks):
                start = i * scaled_cond_size + scaled_block_size
                end = (i + 1) * scaled_cond_size + scaled_block_size
                mask[start:end, start:end] = 0  # Diagonal blocks
            mask = mask * -1e20
            mask = mask.to(query.dtype)

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=mask)
            self.bank_attn = hidden_states[:, :, scaled_block_size:, :]
        
        else:
            query = attn.to_q(hidden_states) 
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
    
            inner_dim = query.shape[-1]
            head_dim = inner_dim // attn.heads
            
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            key = torch.concat([key[:, :, :block_size, :], self.bank_kv[0]], dim=-2)
            value = torch.concat([value[:, :, :block_size, :], self.bank_kv[1]], dim=-2)
            
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            
            query = query[:, :, :scaled_block_size, :]

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=None)
            hidden_states = torch.concat([hidden_states, self.bank_attn], dim=-2)
            
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # Linear projection (with LoRA weight applied to each proj layer)
        hidden_states = attn.to_out[0](hidden_states)
        for i in range(self.n_loras):
             hidden_states = hidden_states + self.lora_weights[i] * self.proj_loras[i](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        cond_hidden_states = hidden_states[:, block_size:,:]
        hidden_states = hidden_states[:, :block_size,:]
        
        return (hidden_states, encoder_hidden_states, cond_hidden_states) if use_cond else (encoder_hidden_states, hidden_states)