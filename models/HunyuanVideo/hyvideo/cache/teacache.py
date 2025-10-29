# coding=utf-8
# Adapted from  
# https://github.com/ali-vilab/TeaCache,
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Copyright (C) 2025 ali-vilab. All rights reserved.
#
# This code is based on ali-vilab's TeaCache library and the TeaCache
# implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to TeaCache used by ali-vilab team.
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
import os
import time
import json
from datetime import datetime
from typing import Any, List, Tuple, Optional, Union, Dict
from pathlib import Path

from loguru import logger
import torch
import numpy as np

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules.modulate_layers import modulate
from hyvideo.modules.attention import attention, parallel_attention, get_cu_seqlens


def is_should_calc(self, img, vec, txt):
    inp = img.clone()
    vec_ = vec.clone()
    txt_ = txt.clone()
    (
        img_mod1_shift,
        img_mod1_scale,
        img_mod1_gate,
        img_mod2_shift,
        img_mod2_scale,
        img_mod2_gate,
    ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
    normed_inp = self.double_blocks[0].img_norm1(inp)
    modulated_inp = modulate(
        normed_inp, shift=img_mod1_shift, scale=img_mod1_scale
    )
    if self.cnt == 0 or self.cnt == self.num_steps - 1:
        should_calc = True
        self.accumulated_rel_l1_distance = 0
    else: 
        coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
        rescale_func = np.poly1d(coefficients)
        self.accumulated_rel_l1_distance += rescale_func(
            ((modulated_inp - self.previous_modulated_input).abs().mean() /
            self.previous_modulated_input.abs().mean()).cpu().item()
        )
        if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
            should_calc = False
        else:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
    self.previous_modulated_input = modulated_inp  
    self.cnt += 1
    if self.cnt == self.num_steps:
        self.cnt = 0
    return should_calc


def add_teacache_class(pipe, args):
    pipe.__class__.enable_teacache = True
    pipe.__class__.cnt = 0
    pipe.__class__.num_steps = args.infer_steps
    pipe.__class__.rel_l1_thresh = args.rel_l1_thresh
    pipe.__class__.accumulated_rel_l1_distance = 0
    pipe.__class__.previous_modulated_input = None
    pipe.__class__.previous_residual = None
    pipe.__class__.teacache_cnt = 0
    pipe.__class__.is_should_calc = is_should_calc
    return pipe