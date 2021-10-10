from typing import Optional, Union, Tuple, Any
from dataclasses import dataclass
import torch


@dataclass
class Tensor:
    shape: Tuple[Union[int, str], ...]
    dtype: Any = None
    device: Any = None


def zeros(*size: int, out: Optional[Tensor]=None, dtype=None, device=None, requires_grad: bool = False) -> Tensor:
    return Tensor(size, dtype=dtype, device=device)


def tensor(data, dtype=None, device=None) -> Tensor:
    return Tensor(7, dtype, device)


torch_defs = {
    torch.zeros: zeros,
    torch.tensor: tensor
}
