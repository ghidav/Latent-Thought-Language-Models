import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from liger_module import LigerRMSNorm, LigerSwiGLUMLP, liger_rotary_pos_emb, LigerCrossEntropyLoss, LigerLayerNorm

@dataclass
class LTMConfig:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.0
    n_prior_layers: Optional[int] = None
    n_cls_tokens: Optional[int] = None
    window_size: int = 256
    use_liger: bool = True
    max_z_len: int = 32
    use_z_pos_emb: bool = True
    padding: bool = False 


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_rotary_emb_single(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:

    # reshape x to match the complex representation
    x_r, x_i = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, x_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, x_r)

    # apply rotation using real numbers
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # flatten last two dimensions
    x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(3)

    return x_out.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: LTMConfig, cross_attention=False, full_attention=False):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.cross_attention = cross_attention
        self.full_attention = full_attention
        self.window_size = args.window_size  # New parameter for fixed window size
        # print(f"Using sliding window attn with window size {self.window_size}")
        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention with causal mask. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        
        if not self.cross_attention:
            self.register_buffer(
                'attn_mask', self.create_sliding_window_mask(args.max_seq_len, self.window_size), persistent=False
            )
        else:
            self.attn_mask = None

        self.max_seq_len = args.max_seq_len
        self.use_z_pos_emb = args.use_z_pos_emb
        
    def create_sliding_window_mask(self, seq_len, window_size):
        # Create a causal mask
        mask = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            mask[i, start:i+1] = 0  # Allow attending to the window
        return mask
    
    def forward(self, x: torch.Tensor, freqs_cos: Optional[torch.Tensor] = None, freqs_sin: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None, 
        freqs_cos_z: Optional[torch.Tensor] = None, freqs_sin_z: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None): 
        bsz, seqlen, _ = x.shape

        # Determine attention type and prepare Q, K, V
        if self.cross_attention and z is not None:
            assert z.shape[0] == bsz and z.shape[-1] == x.shape[-1], "Batch size and embedding dimension must match"
            xq, xk, xv = self.wq(x), self.wk(z), self.wv(z)
            attn_mask = None
        elif self.full_attention:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            attn_mask = None
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            attn_mask = self.attn_mask[:seqlen, :seqlen]  # Adjust mask for current sequence length

        # Reshape for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, -1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, -1, self.n_local_kv_heads, self.head_dim)

        # RoPE
        if not self.cross_attention and freqs_cos is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        elif self.cross_attention:
            if z is not None and freqs_cos_z is not None:
                xk = apply_rotary_emb_single(xk, freqs_cos_z, freqs_sin_z)
            xq = apply_rotary_emb_single(xq, freqs_cos, freqs_sin)

        # Expand keys and values for grouped multi-query attention
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Move heads to the batch dimension
        xq = xq.transpose(1, 2)  # [batch_size, num_heads, seq_len_q, head_dim]
        xk = xk.transpose(1, 2)  # [batch_size, num_heads, seq_len_kv, head_dim]
        xv = xv.transpose(1, 2)  # [batch_size, num_heads, seq_len_kv, head_dim]      
            
        if attn_mask is not None and padding_mask is not None:
            padding_attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(
                (padding_attn_mask == 1) & (padding_attn_mask.transpose(-1, -2) == 1),
                attn_mask,
                float('-inf')
            )
        elif padding_mask is not None:
            expanded_mask = padding_mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len_q, 1]
            xq = xq * expanded_mask
            cross_attn_mask = padding_mask.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, z.shape[1])
            attn_mask = torch.where(
                cross_attn_mask == 1,
                torch.tensor(0.0, device=cross_attn_mask.device),
                torch.tensor(float('-inf'), device=cross_attn_mask.device)
            )
        
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        else:
            # manual implementation following flash_attn
            print("not using flash attn")
            attn_weight = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                attn_weight = attn_weight + attn_mask
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.dropout(attn_weight, p=0.0, train=True)
            output = torch.matmul(attn_weight, xv)

        # Restore original tensor shape and combine heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LTMConfig, use_cross_attention: bool = False, use_full_attention: bool = False):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.use_cross_attention = use_cross_attention

        self.attention = Attention(args, full_attention=use_full_attention)
        if args.use_liger:
            self.attention = torch.compile(self.attention, dynamic=False)
        if self.use_cross_attention:
            self.cross_attention = Attention(args, cross_attention=True)
            if args.use_liger:
                self.cross_attention = torch.compile(self.cross_attention, dynamic=False)

        if args.use_liger:
            self.feed_forward = LigerSwiGLUMLP(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )
            self.ffn_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)
            self.attention_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)
            if self.use_cross_attention:
                self.cross_attention_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)     
            if self.use_cross_attention:
                self.cross_attention_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, z=None, freqs_cos_z=None, freqs_sin_z=None, padding_mask=None):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, padding_mask=padding_mask)
        if self.use_cross_attention and z is not None:
            h = h + self.cross_attention.forward(self.cross_attention_norm(h), freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask=padding_mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class LatentThoughtModel(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: LTMConfig):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        # For multi-layer implementation
        self.max_z_len = params.max_z_len // params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        # add latent z at all layers or just the last layer
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, use_cross_attention=True))

        self.use_liger = params.use_liger
        if params.use_liger: 
            self.norm = LigerRMSNorm(params.dim, eps=params.norm_eps)
            self.ce = LigerCrossEntropyLoss(
                reduction='mean',
                ignore_index=-1
            )
            self.ce_sum = LigerCrossEntropyLoss(
                reduction='sum',
                ignore_index=-1
            )
        else:
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight  # weight-tying for parameter efficiency

        # precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # positional embeddings for latent z
        self.use_z_pos_emb = params.use_z_pos_emb
        if self.use_z_pos_emb:
            interval = self.params.max_seq_len // self.params.max_z_len
            positions_z = torch.arange(0, self.params.max_seq_len, interval).long()
            freqs_cos_z = freqs_cos[positions_z]
            freqs_sin_z = freqs_sin[positions_z]
            self.register_buffer("freqs_cos_z", freqs_cos_z, persistent=False)
            self.register_buffer("freqs_sin_z", freqs_sin_z, persistent=False)
        else:
            self.freqs_cos_z = None
            self.freqs_sin_z = None
            
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight") or pn.endswith("up_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def decoder_forward(self, tokens: torch.Tensor, z: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Reshape z for multi-layer implementation 
        if z is not None:
            z = z.view(_bsz, self.n_layers, self.max_z_len, -1)

        if self.use_z_pos_emb:
            seq_len_z = z.shape[2]
            freqs_cos_z = self.freqs_cos_z[:seq_len_z]
            freqs_sin_z = self.freqs_sin_z[:seq_len_z]
        else:
            freqs_cos_z = None
            freqs_sin_z = None

        for i, layer in enumerate(self.layers):
            if z is not None and layer.use_cross_attention:
                # Use layer-specific latent vectors
                h = layer(h, freqs_cos, freqs_sin, z[:, i, :, :], freqs_cos_z, freqs_sin_z)
            else:
                h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)
        return h

    def decoder_forward_with_hidden(self, tokens: torch.Tensor, z: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, 
    h_before_cross_attention: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
        if z is not None:
            z = z.view(_bsz, self.n_layers, self.max_z_len, -1)

        if self.use_z_pos_emb:
            seq_len_z = z.shape[2] if z is not None else 0
            freqs_cos_z = self.freqs_cos_z[:seq_len_z]
            freqs_sin_z = self.freqs_sin_z[:seq_len_z]
        else:
            freqs_cos_z = None
            freqs_sin_z = None

        save_h = True  # Flag to save h_before_cross_attention only once
        
        if h_before_cross_attention is not None:
            h = h_before_cross_attention
        else:
            h = self.tok_embeddings(tokens)
            h = self.dropout(h)

        for i, layer in enumerate(self.layers):
            if z is not None and layer.use_cross_attention:
                if save_h:
                    h_before_cross_attention = h
                    save_h = False
                # Use layer-specific latent vectors
                h = layer(h, freqs_cos, freqs_sin, z[:, i, :, :], freqs_cos_z, freqs_sin_z, padding_mask=padding_mask)
            else:
                h = layer(h, freqs_cos, freqs_sin, padding_mask=padding_mask)
        h = self.norm(h)
        return h, h_before_cross_attention.detach()

    def forward(self, tokens: torch.Tensor, z: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.decoder_forward(tokens, z, targets)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            if self.use_liger:
                self.last_loss = self.ce(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :])  # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def register_inference_module(self, name, module):
        """Register a learnable inference module so DDP can sync its gradients."""
        self.add_module(f"inference_{name}", module)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, z, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len :]
            logits = self(idx_cond, z)
            logits = logits[:, -1, :]  # crop to just the final time step
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def evaluate_conditional(self, condition_idx, z, candidate_seqs, temperature=1.0):
        # Move condition and z to the same device as the model
        device = next(self.parameters()).device
        condition_idx = condition_idx.to(device)
        z = z.to(device)

        log_likelihoods = []

        # Iterate over each candidate sequence
        for idx, candidate in enumerate(candidate_seqs):
            # Initialize the sequence with the condition
            current_seq = condition_idx.clone()  # Shape: (1, t_condition)
            log_likelihood = 0.0

            # Iterate through each token in the candidate sequence
            for token in candidate[0]:
                with torch.no_grad():
                    logits = self(current_seq, z)  # Assuming logits shape: (1, seq_len, vocab_size)
                    logits = logits[:, -1, :] / temperature  # Get logits for the last token
                    log_probs = F.log_softmax(logits, dim=-1)  # Shape: (1, vocab_size)

                # Extract the log probability of the target token
                token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)  # Shape: (1, 1)
                token_log_prob = log_probs[0, token].item()
                log_likelihood += token_log_prob
                current_seq = torch.cat([current_seq, token_tensor], dim=1)  # Shape: (1, t_condition + t_candidate_so_far)

            log_likelihoods.append(log_likelihood)

        return log_likelihoods

    # VI model functionality for ELBO optimization
    def elbo(
        self,
        tokens: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        eps: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        h_before_cross_attention: Optional[torch.Tensor] = None,
        eval_mode: bool = False,  # Whether to compute perplexity
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:

        _bsz, seqlen = tokens.shape
        weight = 1.0

        # Compute KL divergence
        kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # -KL(q|p)
        kl_loss = kl_div.sum(dim=(1, 2))  # Sum over latent dims, shape = (batch_size,)

        # sample z by reparametrization trick
        z = mu + eps * torch.exp(0.5 * log_var)

        h, h_before_cross_attention = self.decoder_forward_with_hidden(tokens, z, targets, h_before_cross_attention)
        logits = self.output(h)

        if not eval_mode and self.use_liger: # use liger only in training
            nlkhd = self.ce_sum(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            nlkhd = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction="none"
            )
        
        if eval_mode: 
            nlkhd = nlkhd.view(_bsz, -1).sum(dim=-1)  # Sum over sequence length, keep batch dim
            nelbo_per_sample = nlkhd + kl_loss * weight
            perplexity = torch.exp(nelbo_per_sample / seqlen)                
            perplexity = perplexity.mean()  # Average perplexity over batch

            nelbo = nelbo_per_sample.sum()  # Sum over batch
            kl_mean = kl_loss.mean()
            nlkhd_mean = nlkhd.mean()
            return nelbo, perplexity, h_before_cross_attention.detach() if h_before_cross_attention is not None else None, kl_mean, nlkhd_mean

        else: # training
            nlkhd_total = nlkhd.sum()
            kl_loss_total = kl_loss.sum()
            nelbo_total = nlkhd_total + kl_loss_total * weight
            perplexity = None
            return nelbo_total, perplexity, h_before_cross_attention.detach() if h_before_cross_attention is not None else None, kl_loss_total, nlkhd_total