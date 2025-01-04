import torch
import torch.nn as nn
from collections import OrderedDict


class MetaSwiGLUSimpleActivation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        # Initialize beta to a parameter tensor of the same size as intermediate_size
        self.beta = nn.Parameter(torch.ones(self.intermediate_size))

    def forward(self, input):
        # input shape = (batch_size, seq_length, intermediate_size)
        beta_unsqueezed = self.beta.unsqueeze(0).unsqueeze(0)
        return input * torch.sigmoid(beta_unsqueezed * input)




class MetaSwiGLUDataDrivenActivation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.beta_generator = nn.Linear(config.intermediate_size, config.intermediate_size)

    def forward(self, input):
        # input shape = (batch_size, seq_length, intermediate_size)
        beta = torch.sigmoid(self.beta_generator(input))
        return input * torch.sigmoid(beta * input)

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, default_kwargs = content if isinstance(content, tuple) else (content, {})
        return lambda **kwargs: cls(**{**default_kwargs, **kwargs})

class MetaLlamaMLP(nn.Module):
    def __init__(self, config,finetuning_args):
        super().__init__()
        self.finetuning_args = finetuning_args
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        if self.finetuning_args.meta_hidden_act == "meta_swiglu_simple":
            self.meta_act_fn = MetaSwiGLUSimpleActivation(self.config)
        elif self.finetuning_args.meta_hidden_act == "meta_swiglu":
            self.meta_act_fn = MetaSwiGLUDataDrivenActivation(self.config)

    def forward(self, x):
        if isinstance(self.meta_act_fn, MetaSwiGLUDataDrivenActivation) or isinstance(self.meta_act_fn, MetaSwiGLUSimpleActivation):
            down_proj = self.down_proj(self.meta_act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj