from typing import Any

import torch
from torch.utils._pytree import tree_map_only

from sglang.srt.utils.common import is_cuda, is_hip, is_musa, is_npu, is_xpu

if is_cuda() or is_hip() or is_musa() or is_xpu():
    from sgl_kernel import weak_ref_tensor
elif is_npu():
    from torch_npu._C import _weak_ref_tensor as weak_ref_tensor
else:
    raise NotImplementedError(
        "weak_ref_tensor is implemented only for CUDA, XPU, and NPU."
    )


def weak_ref_tensors(tensors: Any) -> Any:
    """Release tensor ownership without changing symbolic/scalar graph outputs."""
    return tree_map_only(torch.Tensor, weak_ref_tensor, tensors)
