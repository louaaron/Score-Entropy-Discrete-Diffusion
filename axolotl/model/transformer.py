import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange
# from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from . import rotary

from torch.nn.attention import SDPBackend, sdpa_kernel

from ..utils_nested import packed_tensor_from_jagged, jagged_from_packed_tensor, coerce_offsets, expand_using_offsets, padded_from_jagged, jagged_from_padded

def modulate(x, shift, scale):

    # print("x modulation", x.shape)
    # print("offsets", x.offsets())
    if scale is not None:
        # print("scale", scale.shape)
        x = x * (1 + scale)
    if shift is not None:
        # print("shift", shift.shape)
        x = x + shift
    
    return x


#################################################################################
#                                  Layers                                       #
#################################################################################


# # old layernorm from original SEDD code
# class LayerNorm(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones([dim]))
#         self.dim = dim
#     def forward(self, x):
#         with torch.amp.autocast('cuda', enabled=False):
#             x = F.layer_norm(x.float(), [self.dim])
#         return x * self.weight[None,None,:]


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, dim, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, force_drop_ids=None):
        use_dropout = self.training and (self.dropout_prob > 0) # If we have a dropout prob and we are in training mode, then use dropout
        if use_dropout or (force_drop_ids is not None): # If we are using dropout or we are forcing dropout
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DiscreteDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm([dim])
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        # self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False) # TODO rename to qkv_proj and out_proj
        # self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm([dim])
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, rotary_cos_sin, c):
        batch_size, max_seq_len = x.shape[0], rotary_cos_sin[0].shape[1]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            if qkv.is_nested:
                cos = coerce_offsets(cos, qkv)
                sin = coerce_offsets(sin, qkv)
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )

        q, k, v = qkv.chunk(3, dim=-3)
        q, k, v = q.squeeze(-3), k.squeeze(-3), v.squeeze(-3)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = F.scaled_dot_product_attention(q, k, v)
        
        x = rearrange(x, 'b h s d -> b s (h d)')

        # out
        x = self.attn_out(x)
        x = modulate(x, None, gate_msa)
        x = self.dropout(x)
        if x_skip.is_nested:
            x_skip = coerce_offsets(x_skip, x) # TODO: should be fixed in a future pytorch release so you don't have to coerce the offsets
        x = x + x_skip

        # mlp operation
        x_skip = x
        x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.mlp(x)
        x = modulate(x, None, gate_mlp)
        x = self.dropout(x)
        if x_skip.is_nested:
            x_skip = coerce_offsets(x_skip, x) # TODO: should be fixed in a future pytorch release so you don't have to coerce the offsets
        x = x + x_skip

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        # # old implementation
        # self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        # torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        # new implementation
        self.embedding = nn.Embedding(vocab_dim, dim)
        torch.nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, x) -> torch.Tensor:

        return self.embedding(x)


class DiscreteDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm([hidden_size])
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        # self.fc = nn.Linear(hidden_size, out_channels) # TODO rename to linear?
        # self.fc.weight.data.zero_()
        # self.fc.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm_final(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x


class DiscreteDiT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config): # TODO: don't use the config object directly, use the parameters
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        self.absorb: bool = config.graph.type == "absorb"
        self.flow: bool = config.graph.type == "flow"
        if self.flow:
            self.absorb = True

        self.prediction_type = config.prediction_type # 'log_score' or 'x0'
        self.dim: int = config.tokens + (1 if self.absorb else 0)
        self.num_labels: int = config.num_labels

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, self.dim)
        self.label_embed = LabelEmbedder(self.num_labels, config.model.cond_dim, config.model.label_dropout)
        self.t_embed = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        self.blocks = nn.ModuleList([
            DiscreteDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout) for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DiscreteDitFinalLayer(config.model.hidden_size, self.dim, config.model.cond_dim)
        self.scale_by_sigma = config.model.scale_by_sigma


    def forward(self, indices, t, label, sigma=None, x1=None): # sigma is only used for SEDD absorb

        # convert the absorb tokens to uniform looking. This is a hack to make it work for now.
        # Input indices look like uniform diffusion, but we still ahve indices, which have the absorb tokens.
        if self.flow:
            assert self.absorb, "Flow is only supported for absorb mode" # TODO: this is a hack to make it work for now
            assert self.prediction_type == 'x0_flow', "Flow is only supported for x0 prediction type" # TODO: this is a hack to make it work for now
            if x1 is None:
                x1 = torch.randint_like(indices, 0, self.dim - 1)
            input_indices = torch.where(indices == self.dim - 1, x1, indices)
        else:
            input_indices = indices
        
        x: torch.Tensor = self.vocab_embed(input_indices)

        time_embed = self.t_embed(t)
        label_embed = self.label_embed(label)
        c = F.silu(time_embed + label_embed)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c)

            x = self.output_layer(x, c)


        if self.prediction_type == 'log_score':
            if self.absorb:
                assert sigma is not None, "Absorb requires sigma to be passed in"
                esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
                x = x - esigm1_log - np.log(x.shape[-1] - 1) # this will be approximately averaged at 0
            
            # Seems to maybe stabilize uniform training. But not sure if this is the right way to do it.
            # indices_mask = F.one_hot(indices, num_classes=self.dim).to(torch.bool)
            # x = torch.where(indices_mask, 0, x)
        
        elif self.prediction_type in ['x0', 'x0_flow']:
            # Make the last dim -inf for the absorb token
            if self.absorb:

                masking_state = torch.ones_like(indices) * (self.dim - 1)
                mask = F.one_hot(masking_state, num_classes=self.dim).to(torch.bool)
                indices_mask = F.one_hot(indices, num_classes=self.dim).to(torch.bool)
                x = torch.where(indices_mask, 10000, x)
                x = torch.where(mask, -torch.inf, x)

                
            elif not self.absorb:
                pass
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented yet!")

        return x
