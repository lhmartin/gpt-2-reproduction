from typing import Literal, get_args
from torch import Tensor, matmul, tril, ones, no_grad
from math import sqrt
import torch
import torch.nn.functional as F
from pydantic import BaseModel
import torch.nn as nn

MODEL_TYPES = Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


class GPT2Model(nn.Module):
    class Config(BaseModel):
        block_size: int = 1024
        vocab_size: int = 50257
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # token embeddings
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # position embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # main trunk
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # Final layer norm
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        # final projection from embddings to vocab size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme. Make the input embeds, and output embeds together
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layers) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type: MODEL_TYPES):
        assert model_type in get_args(MODEL_TYPES)

        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained gpt : {model_type}")

        config = {
            # 124M Params
            "gpt2": GPT2Model.Config(n_layer=12, n_head=12, n_embd=768),
            # 350M Params
            "gpt2-medium": GPT2Model.Config(n_layer=24, n_head=16, n_embd=1024),
            # 774M Params
            "gpt2-large": GPT2Model.Config(n_layer=36, n_head=20, n_embd=1280),
            # 1558M Params
            "gpt2-xl": GPT2Model.Config(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config.vocab_size = 50257
        config.block_size = 1024

        model = GPT2Model(config)

        sd = model.state_dict()

        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"Missmatched keys : {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                assert sd_hf[k].shape == sd[k].shape
                with no_grad():
                    sd[k].copy_(sd_hf[k])

        print("Yay! Model loaded!")
        return model

    def forward(self, idx: Tensor):
        batch_size, seq_len = idx.size()

        assert (
            seq_len <= self.config.block_size
        ), f"Cannot forward sequecne longer than block size {self.config.block_size}"

        # forward the token and positional embeddings
        pos_input = torch.arange(
            0, seq_len, dtype=torch.long, device=idx.device
        )  # (seq_len, n_embd)
        pos_emb = self.transformer.wpe(pos_input)
        tok_emb = self.transformer.wte(idx)

        # combine position and token embeddings
        x = pos_emb + tok_emb

        # go through each block in the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward through the last layer norm
        x = self.transformer.ln_f(x)

        # final classifier
        logits = self.lm_head(x)

        return logits


class Block(nn.Module):
    def __init__(self, config: GPT2Model.Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class MLP(nn.Module):
    def __init__(self, config: GPT2Model.Config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # don't need to use the approx.
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CasualSelfAttention(nn.Module):
    def __init__(self, config: GPT2Model.Config) -> None:
        super().__init__()

        # project to K,V,Q shape
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias",
            tril(ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: Tensor):
        batch_size, seq_len, embed_dim = x.size()

        qkv = self.c_attn(x)
        # split into the the three components
        queries, keys, values = qkv.split(self.n_embd, dim=2)

        queries = self._reshape_to_heads(queries)
        keys = self._reshape_to_heads(keys)
        values = self._reshape_to_heads(values)

        embeddings = self.atn_method(
            queries=queries,
            keys=keys,  # [batch, n_head, seq_len, query_dim]
            values=values,  # [batch, n_head, seq_len, values_dim]
            mask=self.bias[:, :, :seq_len, :seq_len],
        )

        # recombine the heads
        embeddings = (
            embeddings.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        )

        embeddings = self.c_proj(embeddings)

        return embeddings

    def _reshape_to_heads(self, input_matrix: Tensor) -> Tensor:
        """Reshape an input matrix into a 4-dim matrix of shape: [batch_size, n_heads, seq_len, embed_size // n_heads]

        Args:
            input_matrix (Tensor): 3-dim matrix of shape : [batch_size, seq_len, embed_size]

        Returns:
            Tensor: The reshaped array of shape : [batch_size, n_heads, seq_len, embed_size // n_heads]
        """
        # extract the shape info
        bs, seq_len, embed_size = input_matrix.shape

        # split the model embedding size into n_heads
        input_matrix = input_matrix.view(
            bs, seq_len, self.n_head, embed_size // self.n_head
        )

        # swap the head and sequence length dimensions
        return input_matrix.transpose(1, 2)

    def atn_method(
        self,
        queries: Tensor,  # [batch x n_head x seq_len x query_dim]
        keys: Tensor,  # [batch x n_head x seq_len x query_dim]
        values: Tensor,  # [batch x n_head x seq_len x values_dim]
        mask: Tensor | None = None,  # mask out certain input values
    ) -> Tensor:
        # First Q * K -> [batch, n_head, seq_len, seq_len]
        qk = matmul(queries, keys.transpose(-2, -1))

        # Scale
        qv = qk * (1.0 / sqrt(keys.size(-1)))

        # Mask out illegal connections during on the target side or padding on the input side
        if mask is not None:
            qv = qv.masked_fill(mask == 0, float("-inf"))

        # Softmax, ie: scale between [0,1] and distribute importance across the input
        atn_values = F.softmax(qv, dim=-1)

        # Multiple with values, ie: scale values by attention values
        embeddings = matmul(atn_values, values)

        return embeddings


if __name__ == "__main__":
    num_return_sequences = 5
    max_length = 30

    # model = GPT2Model.from_pretrained('gpt2')
    model = GPT2Model(GPT2Model.Config())
    model.eval()
    model.to("cuda")

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    x = tokens.to("cuda")

    print(x)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model.forward(x)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)

            ix = torch.multinomial(topk_probs, 1)

            xcol = torch.gather(topk_indicies, -1, ix)

            x = torch.cat((x, xcol), dim=1).long()

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"> {decoded}")
