import math
import numpy as np
import torch
import torch.nn as nn

from typing import Optional, List, Tuple, Iterable, Callable
from einops import einsum, rearrange

class Linear(nn.Module):
    """Applies a linear transformation to the input: y = xA^T + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the linear module.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If True, includes a bias term. Defaults to False.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return einsum(x, self.weight, "b ... i, j i -> b ... j") + (self.bias if self.bias is not None else 0)


class Embedding(nn.Module):
    """A lookup table that maps indices to embedding vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Looks up embedding vectors for token IDs.

        Args:
            token_ids (torch.Tensor): Input tensor of shape (...).

        Returns:
            torch.Tensor: Output tensor of shape (..., embedding_dim).
        """
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """Applies Root Mean Square Layer Normalization (RMSNorm)."""  

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMSNorm to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        return (x * self.weight) / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    return x * (1 / (1 + torch.exp(-x)))

class SwiGLU(nn.Module):
    """Applies the SwiGLU feedforward transformation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the SwiGLU module.

        Args:
            d_model (int): Hidden dimension of the model.
            d_ff (int): Inner dimension of the feedforward layer.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the SwiGLU transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    """Applies Rotary Position Embeddings (RoPE)."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initializes the RoPE module.

        Args:
            theta (float): Θ value for the rotary embedding.
            d_k (int): Dimension of query and key vectors.
            max_seq_len (int): Maximum sequence length supported.
            device (torch.device, optional): Device to store buffers. Defaults to None.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)
        self.register_buffer('cos_cached', torch.cos(angles))
        self.register_buffer('sin_cached', torch.sin(angles))
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Applies rotary position embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Tensor of shape (..., seq_len)
                specifying token positions.

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_k).
        """
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        result = torch.zeros_like(x)
        result[..., ::2] = rotated_x1
        result[..., 1::2] = rotated_x2
        
        return result

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax activation function.

    Applies the softmax function to the input tensor along the specified dimension.

    Args:
    x: Input tensor.
    dim: Dimension along which softmax will be computed. Defaults to -1.

    Returns:
    Tensor with softmax applied along the specified dimension.
    """
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-softmax activation function.
    """
    x = x - x.max(dim=dim, keepdim=True)[0]
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss function.

    Args:
        inputs (torch.Tensor): Input tensor of shape (..., vocab_size).
        targets (torch.Tensor): Target tensor of shape (...,).

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    return -log_softmax(inputs, dim=-1).gather(dim=-1, index=targets.unsqueeze(-1)).squeeze().mean()

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation function.
    """
    return 1 / (1 + torch.exp(-x))

def tanh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent activation function.
    """
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> torch.Tensor:
    """Clips the gradients of the parameters to have an l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    all_params = torch.cat([param.grad.data.flatten() for param in parameters if param.grad is not None])
    norm = torch.norm(all_params, p=2)
    if norm > max_l2_norm and max_l2_norm > 0:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.div_(norm / max_l2_norm)
    return norm


def _apply_temperature_top_p(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.nn.functional.one_hot(
            logits.argmax(dim=-1), num_classes=logits.shape[-1]
        ).to(dtype=logits.dtype)
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    if top_p < 1.0:
        top_p = max(top_p, 1e-5)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = (cumulative - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs = torch.zeros_like(probs)
        probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    with torch.no_grad():
        probs = _apply_temperature_top_p(logits, temperature, top_p)
        if temperature <= 0:
            token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            token_ids = torch.multinomial(probs, num_samples=1)
    return token_ids

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Returns the learning rate at the given iteration under the specified cosine learning rate schedule with warmup."""
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate) / 2 * (1 + np.cos(np.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    else:
        return min_learning_rate

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Scaled dot-product attention function.

    Args:
        query: Tensor of shape (batch_size, ..., seq_len_q, d_k)
        key: Tensor of shape (batch_size, ..., seq_len_k, d_k)  
        value: Tensor of shape (batch_size, ..., seq_len_v, d_v)
        mask: Boolean tensor of shape (seq_len_q, seq_len_k) or broadcastable shape

    Returns:
        Tensor of shape (batch_size, ..., seq_len_q, d_v)
    """
    attn_score = einsum(query, key, "b ... q d_k, b ... k d_k -> b ... q k") / math.sqrt(query.shape[-1])
    if mask is not None:
        attn_score = attn_score.masked_fill(~mask, float('-inf'))
    attn_prob = softmax(attn_score, dim=-1)
    return einsum(attn_prob, value, "b ... q k, b ... k d_v -> b ... q d_v")

class CasualMultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_rope: bool = False,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        """Initializes the attention module.

        Args:
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            device (torch.device, optional): Device to store parameters. Defaults to None.
        dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
            use_rope (bool, optional): Whether to apply RoPE. Defaults to False.
            theta (float, optional): Θ parameter for RoPE when enabled. Defaults to None.
            max_seq_len (int, optional): Maximum sequence length for RoPE buffers.
                Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if use_rope:
            self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Applies causal multi-head self-attention.

        Args:
        x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).
            token_positions (torch.Tensor, optional): Tensor of shape (..., seq_len)
                with token positions; required if `use_rope` is True. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        q = self.q_proj(x).unflatten(dim=-1, sizes=(self.num_heads, -1)).transpose(-2, -3)
        k = self.k_proj(x).unflatten(dim=-1, sizes=(self.num_heads, -1)).transpose(-2, -3)
        v = self.v_proj(x).unflatten(dim=-1, sizes=(self.num_heads, -1)).transpose(-2, -3)
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(q.shape[-2], device=self.device, dtype=torch.long)
                token_positions = token_positions.expand(*q.shape[:-1])
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        causal_mask = torch.tril(torch.ones(q.shape[-2], k.shape[-2], device=self.device, dtype=torch.bool))
        attn = scaled_dot_product_attention(q, k, v, causal_mask).transpose(-2, -3).flatten(start_dim=-2, end_dim=-1)
        return self.output_proj(attn)

class PointWiseFFN(nn.Module):
    """Point-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))

class TransformerBlock(nn.Module):
    """A single Transformer block with self-attention and feedforward network."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_rope: bool = False,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
        use_point_wise_ffn: bool = False,
    ) -> None:
        """Initializes the Transformer block.

        Args:
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden dimension of the feedforward layer.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
            use_rope (bool, optional): Whether to apply RoPE in self-attention. Defaults to False.
            theta (float, optional): Θ parameter for RoPE. Defaults to None.
            max_seq_len (int, optional): Maximum sequence length for RoPE buffers. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.use_rope = use_rope
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = CasualMultiheadSelfAttention(d_model, num_heads, use_rope=use_rope, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype) if not use_point_wise_ffn else PointWiseFFN(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        x = self.attn(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x

class TransformerLM(nn.Module):
    """A Transformer-based language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_rope: bool = False,
        theta: Optional[float] = None,
        use_point_wise_ffn: bool = False,
    ) -> None:
        """Initializes the Transformer language model.

        Args:
            vocab_size (int): Vocabulary size for token embeddings.
            context_length (int): Maximum sequence length for positional encodings.
            num_layers (int): Number of Transformer blocks.
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden dimension of the feedforward layer.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
            use_rope (bool, optional): Whether to apply RoPE. Defaults to False.
            theta (float, optional): Θ parameter for RoPE. Defaults to None.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.use_rope = use_rope
        self.theta = theta

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, use_rope=use_rope, theta=theta, max_seq_len=context_length, device=device, dtype=dtype, use_point_wise_ffn=use_point_wise_ffn) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Applies the Transformer language model.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (..., seq_len).

        Returns:
            torch.Tensor: Logits of shape (..., seq_len, vocab_size).
        """
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor | list[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        device = self.token_embeddings.weight.device
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        input_tensor = input_ids.to(device)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        generated = input_tensor
        for _ in range(max_new_tokens):
            context = generated[:, -self.context_length :]
            logits = self(context)
            next_logits = logits[:, -1, :]
            next_token = sample_next_token(next_logits, temperature, top_p)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break
        return generated

class LSTMCell(nn.Module):
    """A single Long Short-Term Memory (LSTM) cell."""

    def __init__(
        self,
        d_model: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the LSTM cell.

        Args:
            d_model (int): Hidden dimension of the LSTM.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.weight_ih = nn.Parameter(torch.randn(4 * d_model, d_model, device=device, dtype=dtype))
        self.weight_hh = nn.Parameter(torch.randn(4 * d_model, d_model, device=device, dtype=dtype))
        self.bias_ih = nn.Parameter(torch.randn(4 * d_model, device=device, dtype=dtype))
        self.bias_hh = nn.Parameter(torch.randn(4 * d_model, device=device, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LSTM cell.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_model).
            state (tuple[torch.Tensor, torch.Tensor], optional): Tuple of
                (hidden_state, cell_state), each of shape (batch_size, d_model).
                If None, both are initialized to zeros. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The next (hidden_state, cell_state),
            each of shape (batch_size, d_model).
        """
        if state is None:
            h, c = torch.zeros(x.shape[0], self.d_model, device=self.device, dtype=self.dtype), torch.zeros(x.shape[0], self.d_model, device=self.device, dtype=self.dtype)
        else:
            h, c = state
        x = einsum(x, self.weight_ih, "b d_model, dddd_model d_model -> b dddd_model") + self.bias_ih
        h = einsum(h, self.weight_hh, "b d_model, dddd_model d_model -> b dddd_model") + self.bias_hh
        i = sigmoid(x[..., :self.d_model])
        f = sigmoid(x[..., self.d_model:2*self.d_model])
        o = sigmoid(x[..., 2*self.d_model:3*self.d_model])
        g = tanh(x[..., 3*self.d_model:])
        c = f * c + i * g
        h = o * tanh(c)
        return h, c

class LSTM(nn.Module):
    """Multi-layer LSTM network with batch-first input."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the multi-layer LSTM.

        Args:
            d_model (int): Hidden dimension of the LSTM.
            num_layers (int): Number of stacked LSTM layers.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.lns = nn.ModuleList([RMSNorm(d_model, eps=1e-6, device=device, dtype=dtype) for _ in range(num_layers)])
        self.cells = nn.ModuleList([LSTMCell(d_model, device=device, dtype=dtype) for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Applies the multi-layer LSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            state (tuple[torch.Tensor, torch.Tensor], optional): Tuple of
                (hidden_states, cell_states), each of shape
                (num_layers, batch_size, d_model). Defaults to None.

        Returns:
            tuple:
                - torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
                - tuple[torch.Tensor, torch.Tensor]: Next (hidden_states, cell_states),
                    each of shape (num_layers, batch_size, d_model).
        """
        if state is None:
            h = torch.zeros(self.num_layers, x.shape[0], self.d_model, device=self.device, dtype=self.dtype)
            c = torch.zeros(self.num_layers, x.shape[0], self.d_model, device=self.device, dtype=self.dtype)
        else:
            h, c = state
        result_output = []
        h_new = [h[layer] for layer in range(self.num_layers)]
        c_new = [c[layer] for layer in range(self.num_layers)]
        for t in range(x.shape[1]):
            x_t = x[:, t]
            for layer in range(self.num_layers):
                h_layer, c_layer = self.cells[layer](x_t, (h_new[layer], c_new[layer]))
                x_t = self.lns[layer](h_layer)
                h_new[layer] = h_layer
                c_new[layer] = c_layer
            result_output.append(x_t)
        return torch.stack(result_output, dim=1), (torch.stack(h_new, dim=0), torch.stack(c_new, dim=0))

class LSTMLM(nn.Module):
    """LSTM-based language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        """Initializes the LSTM language model.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Hidden dimension of the LSTM.
            num_layers (int): Number of LSTM layers.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.lstm = LSTM(d_model, num_layers, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Applies the LSTM language model.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len).
            state (tuple[torch.Tensor, torch.Tensor], optional): Tuple of
                (hidden_states, cell_states), each of shape
                (num_layers, batch_size, d_model). Defaults to None.

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        x = self.token_embeddings(input_ids)
        x, (h, c) = self.lstm(x, state)
        logits = self.lm_head(self.ln_final(x))
        return logits, (h, c)

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor | list[int],
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        return_state: bool = False,
    ) -> torch.Tensor:
        self.eval()
        device = self.token_embeddings.weight.device
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        input_tensor = input_ids.to(device)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        generated = input_tensor
        logits, state = self(generated, state)
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            next_token = sample_next_token(next_logits, temperature, top_p)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break
            logits, state = self(next_token, state)
        if return_state:
            return generated, state
        else:
            return generated

class AdamW(torch.optim.Optimizer):
    """AdamW optimizer."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """Initializes the AdamW optimizer."""
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group['params']:
                self.state[param] = {
                    'step': 0,
                    'first_momentum': torch.zeros_like(param),
                    'second_momentum': torch.zeros_like(param)
                }

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                step = state['step']
                first_momentum = state['first_momentum'] * group['betas'][0] + (1 - group['betas'][0]) * grad
                second_momentum = state['second_momentum'] * group['betas'][1] + (1 - group['betas'][1]) * grad * grad
                
                step += 1
                alpha_step = group['lr'] * np.sqrt(1 - group['betas'][1] ** step) / (1 - group['betas'][0] ** step)
                p.data = p.data - alpha_step * first_momentum / (torch.sqrt(second_momentum) + group['eps']) - group['lr'] * group['weight_decay'] * p.data
                state['step'] = step
                state['first_momentum'] = first_momentum
                state['second_momentum'] = second_momentum
        return loss

def get_batch(dataset: np.typing.NDArray, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets a batch of data from the dataset."""
    batch = []
    for _ in range(batch_size):
        start_idx = np.random.randint(0, len(dataset) - context_length)
        batch.append(dataset[start_idx:start_idx + context_length + 1])
    batch = torch.from_numpy(np.array(batch)).long().to(device).clone()
    return batch[:, :-1], batch[:, 1:]

def save_checkpoint(model, optimizer, iteration, out):
    """Saves a checkpoint of the model and optimizer."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def load_checkpoint(src, model, optimizer=None):
    """Loads a checkpoint of the model and optimizer."""
    checkpoint = torch.load(src, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']