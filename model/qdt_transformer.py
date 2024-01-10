import math

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from model.mlp_block import MLP
from model.conv_block import Conv2dNormActivation

import torch
import torch.nn.functional as F

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=torch.nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        
class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6) 
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        
        # attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = torch.nn.MultiheadAttention(hidden_dim, 
                                                          num_heads, 
                                                          dropout=attention_dropout,
                                                          batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        
        # mlp block # no grad?
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        
        
    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expect (batch_size, seq_length, hidden_dim) got {input.shape}")
        
        # attention skip
        x = self.ln_1(input)
        x,_ = self.self_attention(x, x, x, need_weights=False) # why 3 same x?
        x = self.dropout(x)
        x = x + input
        
        # mlp skip
        y = self.ln_2(x)
        y = self.mlp(x)
        
        z = x+y
        
        return z
        
class Encoder(torch.nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        
        self.pos_embedding = torch.nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = torch.nn.Dropout(dropout)
        layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                norm_layer=norm_layer,
            )
        self.layers = torch.nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)
        
    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expect (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., torch.nn.Module] = torch.nn.BatchNorm2d
    activation_layer: Callable[..., torch.nn.Module] = torch.nn.ReLU
    
class QuadTreeTransformer(torch.nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        num_classes: int=1000,
        dropout: float=0.0,
        attention_dropout: float=0.0,
        representation_size: Optional[int] = None,
        norm_layer: Callable[...,torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None, # waht the hell of this?
    ):
        super().__init__()
        torch._assert(image_size%patch_size==0, f"Input shape indivisile by patch size.")
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        
        self.conv_proj = torch.nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        seq_length = (image_size // patch_size) ** 2
        
        # Add a class token
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1
        
        self.encoder = Encoder(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )
        self.seq_length = seq_length
        
        heads_layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = torch.nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = torch.nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = torch.nn.Tanh()
            heads_layers["head"] = torch.nn.Linear(representation_size, num_classes)

        self.heads = torch.nn.Sequential(heads_layers)
        
        if isinstance(self.conv_proj, torch.nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            torch.nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                torch.nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, torch.nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            torch.nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                torch.nn.init.zeros_(self.conv_proj.conv_last.bias)
                
        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, torch.nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            torch.nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            torch.nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, torch.nn.Linear):
            torch.nn.init.zeros_(self.heads.head.weight)
            torch.nn.init.zeros_(self.heads.head.bias)
    
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x
        
    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x