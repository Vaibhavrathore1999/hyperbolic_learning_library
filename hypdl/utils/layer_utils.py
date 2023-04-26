from typing import Callable

from torch.nn import Module

from hypdl.manifolds import Manifold
from hypdl.tensors import ManifoldTensor


def check_if_manifolds_match(layer: Module, input: ManifoldTensor) -> None:
    if layer.manifold != input.manifold:
        raise ValueError(
            f"Manifold of {layer.__class__.__name__} layer is {layer.manifold}"
            f"but input has manifold {input.manifold}"
        )


def check_if_man_dims_match(layer: Module, man_dim: int, input: ManifoldTensor) -> None:
    if man_dim < 0:
        new_man_dim = input.dim() + man_dim
    else:
        new_man_dim = man_dim

    if input.man_dim != new_man_dim:
        raise ValueError(
            f"Layer of type {layer.__class__.__name__} expects the manifold dimension to be {man_dim},"
            f"but input has manifold dimension {input.man_dim}"
        )


def op_in_tangent_space(
    op: Callable, manifold: Manifold, input: ManifoldTensor, dim: int = -1
) -> ManifoldTensor:
    input = manifold.logmap0(input, dim=dim)
    input = op(input)
    return manifold.expmap0(input, dim=dim)
