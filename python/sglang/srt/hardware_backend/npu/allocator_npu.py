from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.kernels.ops.memory.allocator import alloc_decode_kernel
from sglang.srt.utils import get_num_new_pages, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class NPUPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        if num_new_pages is None:
            # Derive the count from the CPU length copies: a device-side sum
            # forces a .item() sync into every prefill step and serializes the
            # launch queue right before the extend kernels.
            num_new_pages_item = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        else:
            num_new_pages_item = num_new_pages
        if num_new_pages_item > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages_item > len(self.free_pages):
            return None

        if num_new_pages_item < 200:
            from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel

            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            max_num_extend_tokens = next_power_of_2(extend_num_tokens)
            bs = prefix_lens.shape[0]
            alloc_extend_kernel[(bs,)](
                prefix_lens,
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                next_power_of_2(bs),
                self.page_size,
                max_num_extend_tokens,
            )

        else:
            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int32,
                device=self.device,
            )
            alloc_extend_naive(
                prefix_lens,
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                self.page_size,
                self.device,
            )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages_item:]
        return out_indices.int()

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )

        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None

        # One triton launch replaces the eager chain (eq -> cast -> cumsum ->
        # cast -> sub -> clamp -> where/index -> mul/add). The cumsum member of
        # that chain dispatches to the AiCPU engine on Ascend, which serializes
        # against AiCore work; the kernel computes the same page starts on
        # device exactly like the CUDA paged allocator does.
        out_indices = torch.empty((seq_lens.shape[0],), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(seq_lens.shape[0],)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(seq_lens.shape[0]),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices.int()

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        # The previous NPU override round-tripped every free through the CPU
        # (blocking D2H copy + host torch.unique + H2D copy back), which put a
        # full-device sync into the step's free path right where the next
        # prefill is submitted (profile: EVENT_WAIT ~4.6ms per step absorbing
        # the drained launch queue). The paged parent does the identical page
        # census entirely on device -- unique -> staged pages, consolidated
        # lazily by merge_and_sort_free at the next allocation-pressure point.
        # Only cross-device inputs are moved here; an H2D never blocks on
        # device progress.
        if free_index.device.type != torch.device(self.device).type:
            free_index = free_index.to(torch.device(self.device), non_blocking=True)
        super().free(free_index)
