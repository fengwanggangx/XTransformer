import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    def __init__(self, d_embed, vocab_size, padding_idx=None):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_embed, padding_idx=padding_idx)
        self.embedding_dim = d_embed

    def forward(self, x):
        return self.word_embeddings(x) * math.sqrt(self.embedding_dim)

    def get_input_embeddings(self):
        return self.word_embeddings


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_head, base=10000.0):
        super(RotaryPositionalEmbedding, self).__init__()
        if d_head % 2 != 0:
            raise ValueError(f"d_head must be even for RoPE, got {d_head}")
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device, dtype, offset=0):
        position = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(position, self.inv_freq.to(device))
        cos = freqs.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin


def apply_rotary_pos_emb(query, key, cos, sin):
    q_even, q_odd = query[..., 0::2], query[..., 1::2]
    k_even, k_odd = key[..., 0::2], key[..., 1::2]

    query = torch.stack(
        (q_even * cos - q_odd * sin, q_even * sin + q_odd * cos),
        dim=-1,
    ).flatten(-2)
    key = torch.stack(
        (k_even * cos - k_odd * sin, k_even * sin + k_odd * cos),
        dim=-1,
    ).flatten(-2)
    return query, key


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_embed, p_dropout=0.1, debug=False):
        super(MultiHeadAttention, self).__init__()
        if d_embed % n_head != 0:
            raise ValueError(f"d_embed must be divisible by n_head, got d_embed={d_embed}, n_head={n_head}")
        self.n_head = n_head
        self.d_head = d_embed // n_head
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head must be even for RoPE, got {self.d_head}")
        self.linears = clones(nn.Linear(d_embed, d_embed), 4)
        self.rope = RotaryPositionalEmbedding(self.d_head)
        self.attn = None
        self.debug = debug
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, query, key, value, mask=None, past_key_value=None, use_cache=False):
        batch_size = query.size(0)
        query_len = query.size(1)
        past_len = 0
        if past_key_value is not None:
            if len(past_key_value) < 2:
                raise ValueError("past_key_value must contain at least key and value tensors")
            past_key, past_value = past_key_value[:2]
            past_len = past_key.size(-2)
        else:
            past_key, past_value = None, None

        q, k, v = [l(x).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        cos, sin = self.rope(q.size(-2), q.device, q.dtype, offset=past_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key is not None:
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)

        key_len = k.size(-2)
        if mask is not None:
            expected_shape = (batch_size, query_len, key_len)
            if mask.dim() != 3 or tuple(mask.shape) != expected_shape:
                raise ValueError(
                    "mask must have shape "
                    f"[batch_size, query_len, key_len] = {expected_shape}, "
                    f"got {tuple(mask.shape)}"
                )
            mask = mask.unsqueeze(1)

        x, p_attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        self.attn = p_attn.detach() if self.debug else None
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_head)
        x = self.linears[-1](x)
        if use_cache:
            return x, (k, v)
        return x

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(features, eps=eps)

    def forward(self, x):
        return self.norm(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_embed, d_ff)
        self.w_2 = nn.Linear(d_ff, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask, past_key_value=None, use_cache=False):
        if use_cache:
            norm_x = self.sublayer[0].norm(x)
            attn_out, present_key_value = self.self_attn(
                norm_x,
                norm_x,
                norm_x,
                mask,
                past_key_value=past_key_value,
                use_cache=True,
            )
            x = x + self.sublayer[0].dropout(attn_out)
            x = self.sublayer[1](x, self.feed_forward)
            return x, present_key_value

        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, past_key_values=None, use_cache=False):
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError(
                f"past_key_values must have {len(self.layers)} layers, got {len(past_key_values)}"
            )

        present_key_values = [] if use_cache else None
        for layer_index, layer in enumerate(self.layers):
            past_key_value = past_key_values[layer_index] if past_key_values is not None else None
            if use_cache:
                x, present_key_value = layer(
                    x,
                    mask,
                    past_key_value=past_key_value,
                    use_cache=True,
                )
                present_key_values.append(present_key_value)
            else:
                x = layer(x, mask)

        x = self.norm(x)
        if use_cache:
            return x, tuple(present_key_values)
        return x

class Generator(nn.Module):
    def __init__(self, d_embed, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_embed, vocab, bias=False)

    def forward(self, x):
        return self.proj(x)

def padding_mask(input_ids, pad=None):
    pad = 0 if pad is None else pad
    return (input_ids != pad).unsqueeze(-2)


def key_padding_mask(input_ids, pad=None):
    pad = 0 if pad is None else pad
    return input_ids != pad


def causal_mask(size, device):
    return torch.tril(torch.ones(1, size, size, dtype=torch.bool, device=device))


def make_std_mask(input_ids, pad=None):
    return padding_mask(input_ids, pad) & causal_mask(input_ids.size(-1), input_ids.device)


def normalize_past_key_padding_mask(past_key_padding_mask, batch_size, past_len, device):
    if past_len == 0:
        return None
    if past_key_padding_mask is None:
        return torch.ones(batch_size, past_len, dtype=torch.bool, device=device)

    if past_key_padding_mask.dim() == 3 and past_key_padding_mask.size(1) == 1:
        past_key_padding_mask = past_key_padding_mask.squeeze(1)
    expected_shape = (batch_size, past_len)
    if past_key_padding_mask.dim() != 2 or tuple(past_key_padding_mask.shape) != expected_shape:
        raise ValueError(
            "past_key_padding_mask must have shape "
            f"[batch_size, past_len] = {expected_shape}, got {tuple(past_key_padding_mask.shape)}"
        )
    return past_key_padding_mask.to(device=device, dtype=torch.bool)


def combine_key_padding_mask(input_ids, past_len=0, pad=None, past_key_padding_mask=None):
    batch_size = input_ids.size(0)
    current_padding = key_padding_mask(input_ids, pad)
    past_padding = normalize_past_key_padding_mask(
        past_key_padding_mask,
        batch_size,
        past_len,
        input_ids.device,
    )
    if past_len > 0:
        return torch.cat([past_padding, current_padding], dim=-1)
    return current_padding


def make_cache_mask(input_ids, past_len, pad=None, past_key_padding_mask=None):
    current_len = input_ids.size(-1)
    padding = combine_key_padding_mask(
        input_ids,
        past_len=past_len,
        pad=pad,
        past_key_padding_mask=past_key_padding_mask,
    ).unsqueeze(1)
    current_causal = causal_mask(current_len, input_ids.device)
    if past_len > 0:
        past_visible = torch.ones(
            1,
            current_len,
            past_len,
            dtype=torch.bool,
            device=input_ids.device,
        )
        causal = torch.cat([past_visible, current_causal], dim=-1)
    else:
        causal = current_causal

    return padding & causal

class Transformer(nn.Module):
    def __init__(self, decoder, input_embed, generator, padding_idx, max_seq_len):
        super().__init__()
        self.decoder = decoder
        self.input_embed = input_embed
        self.generator = generator
        self.padding_idx = padding_idx
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, mask=None, past_key_values=None, use_cache=False):
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        current_len = input_ids.size(-1)
        if current_len <= 0:
            raise ValueError("input_ids sequence length must be greater than 0")
        past_len = 0
        past_key_padding_mask = None
        if past_key_values is not None:
            if len(past_key_values) == 0:
                raise ValueError("past_key_values must not be empty")
            past_len = past_key_values[0][0].size(-2)
            if len(past_key_values[0]) >= 3:
                past_key_padding_mask = past_key_values[0][2]

        total_len = past_len + current_len
        if total_len > self.max_seq_len:
            raise ValueError(
                f"sequence length exceeds max_seq_len: past_len={past_len}, "
                f"current_len={current_len}, max_seq_len={self.max_seq_len}"
            )

        if mask is None:
            if past_len > 0:
                mask = make_cache_mask(
                    input_ids,
                    past_len,
                    self.padding_idx,
                    past_key_padding_mask=past_key_padding_mask,
                )
            else:
                mask = make_std_mask(input_ids, self.padding_idx)

        hidden_states = self.input_embed(input_ids)
        if use_cache:
            hidden_states, present_key_values = self.decoder(
                hidden_states,
                mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            present_key_padding_mask = combine_key_padding_mask(
                input_ids,
                past_len=past_len,
                pad=self.padding_idx,
                past_key_padding_mask=past_key_padding_mask,
            )
            present_key_values = tuple(
                (present_key, present_value, present_key_padding_mask)
                for present_key, present_value in present_key_values
            )
            return self.generator(hidden_states), present_key_values

        hidden_states = self.decoder(hidden_states, mask)
        return self.generator(hidden_states)

    def get_input_embeddings(self):
        for module in self.input_embed.modules():
            if module is self.input_embed:
                continue
            if hasattr(module, "get_input_embeddings"):
                return module.get_input_embeddings()
        raise RuntimeError("input embedding module does not expose get_input_embeddings()")


def zero_padding_embedding(model, padding_idx=None):
    padding_idx = getattr(model, "padding_idx", 0) if padding_idx is None else padding_idx
    with torch.no_grad():
        model.get_input_embeddings().weight[padding_idx].zero_()


def make_model(cfg, vocab_size=None, N=None, d_embed=None,
               d_ff=None, h=None, dropout=None):
    vocab_size = vocab_size if vocab_size is not None else cfg.model.vocab_size
    N = N if N is not None else cfg.model.n_layers
    d_embed = d_embed if d_embed is not None else cfg.model.d_model
    d_ff = d_ff if d_ff is not None else cfg.model.d_ff
    h = h if h is not None else cfg.model.n_heads
    dropout = dropout if dropout is not None else cfg.model.dropout
    padding_idx = cfg.tokens.padding_idx
    max_seq_len = cfg.model.max_seq_len

    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_embed, dropout, debug=cfg.runtime.debug)
    ff = PositionwiseFeedForward(d_embed, d_ff, dropout)

    decoder_layer = DecoderLayer(d_embed, c(attn), c(ff), dropout)
    decoder = Decoder(decoder_layer, N)
    word_embed = Embeddings(d_embed, vocab_size, padding_idx=padding_idx)
    input_embed = nn.Sequential(word_embed, nn.Dropout(dropout))
    generator = Generator(d_embed, vocab_size)
    generator.proj.weight = word_embed.word_embeddings.weight

    model = Transformer(
        decoder,
        input_embed,
        generator,
        padding_idx=padding_idx,
        max_seq_len=max_seq_len,
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    zero_padding_embedding(model)

    return model.to(cfg.device)
