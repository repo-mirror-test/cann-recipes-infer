# coding=utf-8
# Adapted from
# https://modelers.cn/models/MindIE/Wan2.2/blob/main/wan/modules/attn_layer.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import os
from typing import Any

import torch
from torch import Tensor
import torch_npu
import torch.distributed as dist
from yunchang import LongContextAttention

try:
    from yunchang.kernels import AttnType
except ImportError as e:
    raise ImportError("Please install yunchang 0.6.0 or later") from e

from .attention import attention
from ..distributed.parallel_mgr import get_sp_group
from ..distributed.comm import all_to_all_4D


logger = logging.getLogger(__name__)
MAX_TOKEN = 2147483647


class xFuserLongContextAttention(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        args: Any,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        attn_type: AttnType = AttnType.FA,
    ) -> None:
        """
        Arguments:
            scatter_idx: int = 2, the scatter dimension index for Ulysses All2All
            gather_idx: int = 1, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
        """
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            attn_type=attn_type,
        )
        self.use_kv_cache = use_kv_cache
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )
        self.world_size = dist.get_world_size()
        self.args = args
        self.video_size = ['480*832', '832*480', '480*720', '720*480']

        self.algo = int(os.getenv('ALGO', 0))

        if self.args.size in self.video_size:
            self.use_all_head = True
        else:
            self.use_all_head = False
        
        self.ulysses_pg = get_sp_group().ulysses_group
        self.ring_pg = get_sp_group().ring_group


    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        seq_lens: int,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
        scale=None,
        t_idx=-1,
    ) -> Tensor:
        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, 
            a replicated tensor among processes appended to the front or rear of query, 
            depends the joint_strategy  
            joint_tensor_key: Tensor = None, 
            a replicated tensor among processes appended to the front or rear of key, 
            depends the joint_strategy
            joint_tensor_value: Tensor = None, 
            a replicated tensor among processes appended to the front or rear of value, 
            depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", 
            the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """

        query = all_to_all_4D(input_=query, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        key = all_to_all_4D(input_=key, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        value = all_to_all_4D(input_=value, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

        if get_sp_group().ring_world_size > 1:
            ring_size = get_sp_group().ring_world_size
            b, s, n, d = key.shape
            k_full = torch.empty([ring_size, b, s, n, d], dtype=query.dtype, device=query.device)
            dist.all_gather_into_tensor(k_full, key, group=self.ring_pg)
            key = k_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)

            v_full = torch.empty([ring_size, b, s, n, d], dtype=query.dtype, device=query.device)
            dist.all_gather_into_tensor(v_full, value, group=self.ring_pg)
            value = v_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)

        ori_seqlen = query.shape[1]
        if seq_lens is not None and seq_lens < ori_seqlen:
            query_layer, query_pad = query[:, :seq_lens, :, :], query[:, seq_lens:, :, :]
            key_layer, key_pad = key[:, :seq_lens, :, :], key[:, seq_lens:, :, :]
            value_layer, value_pad = value[:, :seq_lens, :, :], value[:, seq_lens:, :, :]
        else:
            query_layer, key_layer, value_layer = query, key, value

        if self.use_all_head:
            if self.algo == 0 or self.algo == 1:
                out = attention(
                    q=query_layer,
                    k=key_layer,
                    v=value_layer,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale or scale,
                    causal=causal,
                    window_size=window_size,
                    deterministic=deterministic,
                )
            else:
                raise ValueError(f"select flash attention algorithm only support 0, 1, but got {self.algo}")
        else:
            query_layer_list = query_layer.split(1, dim=2)
            key_layer_list = key_layer.split(1, dim=2)
            value_layer_list = value_layer.split(1, dim=2)
            output = []
            for_loop = query_layer.shape[2]
            for i in range(for_loop):
                if self.algo == 0 or self.algo == 1:
                    out = attention(
                        q=query_layer_list[i],
                        k=key_layer_list[i],
                        v=value_layer_list[i],
                        dropout_p=dropout_p,
                        softmax_scale=softmax_scale or scale,
                        causal=causal,
                        window_size=window_size,
                        deterministic=deterministic,
                    )
                else:
                    raise ValueError(f"select flash attention algorithm only support 0, 1, but got {self.algo}")
                output.append(out)
            out = torch.cat(output, dim=2)
        
        if seq_lens is not None and seq_lens < ori_seqlen:
            out_pad = attention(
                q=query_pad,
                k=key_pad, 
                v=value_pad,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale or scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
            out = torch.cat([out, out_pad], dim=1)

        if isinstance(out, tuple):
            context_layer, _, _ = out
        else:
            context_layer = out

        output = all_to_all_4D(input_=context_layer, scatter_idx=1, gather_idx=2, group=self.ulysses_pg)

        return output

