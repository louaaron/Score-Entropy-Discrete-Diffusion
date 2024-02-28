import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)

@torch.jit.script
def modulate_fused(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return modulate(x, shift, scale)