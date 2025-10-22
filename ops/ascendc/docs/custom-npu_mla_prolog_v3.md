# custom-npu_mla_prolog_v3

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Atlas A3 推理系列产品|      √     |

## 功能说明

推理场景下Multi-Head Latent Attention前处理的计算操作。

该算子实现四条并行的计算路径：

1. **标准Query路径**：输入$x$ → $W^{DQ}$下采样 → RmsNorm → $W^{UQ}$上采样 → $W^{UK}$上采样 → $q^N$
2. **位置编码Query路径**：输入$x$ → $W^{DQ}$下采样 → RmsNorm → $W^{QR}$ → ROPE旋转位置编码 → $q^R$
3. **标准Key路径**：输入$x$ → $W^{DKV}$下采样 → RmsNorm → Cache存储 → $k^C$
4. **位置编码Key路径**：输入$x$ → $W^{KR}$ → ROPE旋转位置编码 → Cache存储 → $k^R$

### 计算公式

#### RmsNorm公式

$$
\text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
$$

$$
\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
$$

#### 路径1：标准Query计算

包括下采样、RmsNorm和两次上采样：

$$
c^Q = RmsNorm(x \cdot W^{DQ})
$$

$$
q^C = c^Q \cdot W^{UQ}
$$

$$
q^N = q^C \cdot W^{UK}
$$

#### 路径2：位置编码Query计算

对Query进行ROPE旋转位置编码：

$$
q^R = ROPE(c^Q \cdot W^{QR})
$$

#### 路径3：标准Key计算

包括下采样、RmsNorm，将计算结果存入Cache：

$$
c^{KV} = RmsNorm(x \cdot W^{DKV})
$$

$$
k^C = Cache(c^{KV})
$$

#### 路径4：位置编码Key计算

对Key进行ROPE旋转位置编码，并将结果存入Cache：

$$
k^R = Cache(ROPE(x \cdot W^{KR}))
$$


## 函数原型
```
custom.npu_mla_prolog_v3(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor cache_index, Tensor(a!) kv_cache, Tensor(b!) kr_cache, *, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, Tensor? actual_seq_len=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode='PA_BSND', int query_norm_flag=0, int weight_quant_mode=0, int kv_quant_mode=0, int query_quant_mode=0, int ckvkr_repo_mode=0, int quant_scale_repo_mode=0, int tile_size=128, float k_nope_clip_alpha=1.0, float qc_qr_scale=1.0, float kc_scale=1.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

>**说明：**<br> 
>
>- B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、He（Head Size）表示隐藏层大小、N（Head Num）表示多头数、Hcq表示q低秩矩阵维度、Hckv表示kv低秩矩阵维度、D表示qk不含位置编码维度、Dr表示qk位置编码维度、Nkv表示kv的head数、BlockNum表示PagedAttention场景下的块数、BlockSize表示PagedAttention场景下的块大小、T表示BS合轴后的大小。

-   **token_x**（`Tensor`）：必选参数，公式中用于计算Query和Key的输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`、`int8`，shape为[T, He]或[B, S, He]。

-   **weight_dq**（`Tensor`）：必选参数，公式中用于计算Query的下采样权重矩阵$W^{DQ}$。不支持非连续，数据格式支持FRACTAL_NZ，数据类型支持`bfloat16`、`int8`，shape为[He, Hcq]。

-   **weight_uq_qr**（`Tensor`）：必选参数，公式中用于计算Query的上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$。不支持非连续，数据格式支持FRACTAL_NZ，数据类型支持`bfloat16`、`int8`，shape为[Hcq, N*(D+Dr)]。dtype为INT8（量化场景）时：1. 需为per-tensor量化输入；2. 非量化输出时必传dequant_scale_w_uq_qr；3. 量化输出时必传dequant_scale_w_uq_qr、quant_scale_ckv、quant_scale_ckr；4. smooth_scales_cq可选传。dtype为BFLOAT16（非量化场景）时：dequant_scale_w_uq_qr、quant_scale_ckv、quant_scale_ckr、smooth_scales_cq必须传空指针。

-   **weight_uk**（`Tensor`）：必选参数，公式中用于计算Key的上采样权重$W^{UK}$。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[N, D, Hckv]。

-   **weight_dkv_kr**（`Tensor`）：必选参数，公式中用于计算Key的下采样权重矩阵$W^{DKV}$和位置编码权重矩阵$W^{KR}$。不支持非连续，数据格式支持FRACTAL_NZ，数据类型支持`bfloat16`、`int8`，shape为[He, Hckv+Dr]。

-   **rmsnorm_gamma_cq**（`Tensor`）：必选参数，计算$c^Q$的RmsNorm公式中的$\gamma$参数。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[Hcq]。

-   **rmsnorm_gamma_ckv**（`Tensor`）：必选参数，计算$c^{KV}$的RmsNorm公式中的$\gamma$参数。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[Hckv]。

-   **rope_sin**（`Tensor`）：必选参数，用于计算旋转位置编码的正弦参数矩阵。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[T, Dr]或[B, S, Dr]，支持B=0,S=0,T=0的空Tensor。

-   **rope_cos**（`Tensor`）：必选参数，用于计算旋转位置编码的余弦参数矩阵。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[T, Dr]或[B, S, Dr]，支持B=0,S=0,T=0的空Tensor。

-   **cache_index**（`Tensor`）：必选参数，用于存储kv_cache和kr_cache的索引。不支持非连续，数据格式支持ND，数据类型支持`int64`，shape为[T]或[B, S]，支持B=0,S=0,T=0的空Tensor，取值范围需在[0, BlockNum*BlockSize)内。

-   **kv_cache**（`Tensor`）：必选参数，用于cache索引的aclTensor，计算结果原地更新（对应公式中的$k^C$）。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`、`int8`，shape为[BlockNum, BlockSize, Nkv, Hckv]，支持B=0,Skv=0的空Tensor；Nkv与N关联，N是超参，故Nkv不支持dim=0。

-   **kr_cache**（`Tensor`）：必选参数，用于key位置编码的cache，计算结果原地更新（对应公式中的$k^R$）。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`、`int8`，shape为[BlockNum, BlockSize, Nkv, Dr]，支持B=0,Skv=0的空Tensor；Nkv与N关联，N是超参，故Nkv不支持dim=0。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **dequant_scale_x**（`Tensor`，可选）：预留参数，当前版本暂未使用。

-   **dequant_scale_w_dq**（`Tensor`，可选）：预留参数，当前版本暂未使用。

-   **dequant_scale_w_uq_qr**（`Tensor`，可选）：用于MatmulQcQr矩阵乘后反量化操作的per-channel参数。不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[1, N*(D+Dr)]。

-   **dequant_scale_w_dkv_kr**（`Tensor`，可选）：预留参数，当前版本暂未使用。

-   **quant_scale_ckv**（`Tensor`，可选）：用于对kv_cache输出数据做量化操作的参数。不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[1, Hckv]，支持非空Tensor（仅INT8 dtype量化输出场景需传）。

-   **quant_scale_ckr**（`Tensor`，可选）：用于对kr_cache输出数据做量化操作的参数。不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[1, Dr]，支持非空Tensor（仅INT8 dtype量化输出场景需传）。

-   **smooth_scales_cq**（`Tensor`，可选）：用于对RmsNormCq输出做动态量化操作的参数。不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[1, Hcq]或[1]，支持非空Tensor（仅INT8 dtype场景可选传）。

-   **actual_seq_len**（`Tensor`，可选）：预留参数，当前版本暂未使用。

-   **rmsnorm_epsilon_cq**（`float`，可选）：计算$c^Q$的RmsNorm公式中的$\epsilon$参数，Host侧参数。用户未特意指定时，建议传入1e-05，仅支持double类型，默认值为1e-05。

-   **rmsnorm_epsilon_ckv**（`float`，可选）：计算$c^{KV}$的RmsNorm公式中的$\epsilon$参数，Host侧参数。用户未特意指定时，建议传入1e-05，仅支持double类型，默认值为1e-05。

-   **cache_mode**（`str`，可选）：表示kv_cache的模式，Host侧参数。用户未特意指定时，建议传入"PA_BSND"，仅支持char*类型，可选值为"PA_BSND"、"PA_NZ"，默认值为'PA_BSND'。

-   **query_norm_flag**（`int`，可选）：表示是否输出queryNorm，Host侧参数。仅支持bool类型，0表示不输出queryNorm，1表示输出queryNorm，默认值为0。

-   **weight_quant_mode**（`int`，可选）：表示weight_dq、weight_uq_qr、weight_uk、weight_dkv_kr的量化模式，Host侧参数。仅支持int64类型，0表示非量化，1表示weight_uq_qr量化，默认值为0。

-   **kv_quant_mode**（`int`，可选）：表示kv_cache的量化模式，Host侧参数。仅支持int64类型，0表示非量化，2表示per-channel量化，默认值为0。

-   **query_quant_mode**（`int`，可选）：预留参数，默认值为0。

-   **ckvkr_repo_mode**（`int`，可选）：预留参数，默认值为0。

-   **quant_scale_repo_mode**（`int`，可选）：预留参数，默认值为0。

-   **tile_size**（`int`，可选）：预留参数，默认值为128。

-   **k_nope_clip_alpha**（`float`，可选）：预留参数，默认值为1.0。

-   **qc_qr_scale**（`float`，可选）：预留参数，默认值为1.0。

-   **kc_scale**（`float`，可选）：预留参数，默认值为1.0。

## 返回值说明
-   **query_out**（`Tensor`）：公式中Query的输出tensor（对应$q^N$）。数据格式支持ND，数据类型支持`bfloat16`、`int8`，shape为[T, N, Hckv]或[B, S, N, Hckv]。

-   **query_rope_out**（`Tensor`）：公式中Query位置编码的输出tensor（对应$q^R$）。数据格式支持ND，数据类型支持`bfloat16`，shape为[T, N, Dr]或[B, S, N, Dr]。

-   **dequant_scale_q_nope**（`Tensor`）：Query输出的反量化参数。数据格式支持ND，数据类型支持`float`，shape为[T]或[B, S]。

-   **query_norm**（`Tensor`）：Query做RmsNormCq后的输出tensor（对应$q^C$）。数据格式支持ND，数据类型支持`bfloat16`、`int8`，shape为[T, Hcq]或[B, S, Hcq]。

-   **dequant_scale_q_norm**（`Tensor`）：Query做RmsNormCq后的反量化参数。数据格式支持ND，数据类型支持`float`，shape为[T]或[B*S]。

## 约束说明
-  shape 格式字段含义说明
    | 字段名       | 英文全称/含义                  | 取值规则与说明                                                                 |
    |--------------|--------------------------------|------------------------------------------------------------------------------|
    | B            | Batch（输入样本批量大小）      | 取值范围：0~65536                                                           |
    | S            | Seq-Length（输入样本序列长度） | 取值范围：0~16                                                              |
    | He           | Head-Size（隐藏层大小）        | 取值固定为：7168、7680                                                            |
    | Hcq          | q 低秩矩阵维度                 | 取值固定为：1536                                                           |
    | N            | Head-Num（多头数）             | 取值范围：1、2、4、8、16、32、64、128                                       |
    | Hckv         | kv 低秩矩阵维度                | 取值固定为：512                                                             |
    | D            | qk 不含位置编码维度            | 取值固定为：128                                                             |
    | Dr           | qk 位置编码维度                | 取值固定为：64                                                              |
    | Nkv          | kv 的 head 数                  | 取值固定为：1                                                               |
    | BlockNum     | PagedAttention 场景下的块数    | 取值为计算 `B*Skv/BlockSize` 的结果后向上取整（Skv 表示 kv 的序列长度，允许取 0） |
    | BlockSize    | PagedAttention 场景下的块大小  | 取值范围：16、128                                                           |
    | T            | BS 合轴后的大小                | 取值范围：0~1048576；注：若采用 BS 合轴，此时 token_x、rope_sin、rope_cos 均为 2 维，cache_index 为 1 维，query_out、query_rope_out 为 3 维 |

- 支持场景：
  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">场景</th>
      <th>含义</th>
    </tr>
    <tr>
      <td colspan="2">非量化</td>
      <td>
          入参：所有入参皆为非量化数据 <br> 
          出参：所有出参皆为非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="2">部分量化</td>
      <td>kv_cache非量化 </td>
      <td>
          入参：weight_uq_qr传入pertoken量化数据，其余入参皆为非量化数据 <br> 
          出参：所有出参返回非量化数据 
      </td>
    </tr>
    <tr>
      <td>kv_cache量化 </td>
      <td> 
          入参：weight_uq_qr传入pertoken量化数据，kv_cache、kr_cache传入perchannel量化数据，其余入参皆为非量化数据 <br> 
          出参：kv_cache、kr_cache返回perchannel量化数据，其余出参返回非量化数据 
      </td>
    </tr>
  </table>

- 在不同量化场景下，参数的dtype和shape组合需要满足如下条件：
  <div style="overflow-x: auto; width: 100%;">
  <table style="table-layout: auto;" border="1">
    <tr>
      <th rowspan="3">参数名</th>
      <th rowspan="2" colspan="2">非量化场景</th>
      <th colspan="4">部分量化场景</th>
    </tr>
    <tr>
      <th colspan="2">kv_cache非量化</th>
      <th colspan="2">kv_cache量化</th>
    </tr>
    <tr>
      <th>dtype</th>
      <th>shape</th>
      <th>dtype</th>
      <th>shape</th>
      <th>dtype</th>
      <th>shape</th>
    </tr>
    <tr>
      <td>token_x</td>
      <td>BFLOAT16</td>
      <td>· (B,S,He) <br> · (T, He)</td>
      <td>BFLOAT16</td>
      <td>· (B,S,He) <br> · (T, He)</td>
      <td>BFLOAT16</td>
      <td>· (B,S,He) <br> · (T, He)</td>
    </tr>
    <tr>
      <td>weight_dq</td>
      <td>BFLOAT16</td>
      <td> (He, Hcq)</td>
      <td>BFLOAT16</td>
      <td> (He, Hcq)</td>
      <td>BFLOAT16</td>
      <td> (He, Hcq)</td>
    </tr>
    <tr>
      <td>weight_uq_qr</td>
      <td>BFLOAT16</td>
      <td> (Hcq, N*(D+Dr))</td>
      <td>INT8</td>
      <td> (Hcq, N*(D+Dr))</td>
      <td>INT8</td>
      <td> (Hcq, N*(D+Dr))</td>
    </tr>
    <tr>
      <td>weight_uk</td>
      <td>BFLOAT16</td>
      <td> (N, D, Hckv)</td>
      <td>BFLOAT16</td>
      <td> (N, D, Hckv)</td>
      <td>BFLOAT16</td>
      <td> (N, D, Hckv)</td>
    </tr>
    <tr>
      <td>weight_dkv_kr</td>
      <td>BFLOAT16</td>
      <td> (He, Hckv+Dr)</td>
      <td>BFLOAT16</td>
      <td> (He, Hckv+Dr)</td>
      <td>BFLOAT16</td>
      <td> (He, Hckv+Dr)</td>
    </tr>
    <tr>
      <td> rmsnorm_gamma_cq </td>
      <td>BFLOAT16</td>
      <td> (Hcq)</td>
      <td>BFLOAT16</td>
      <td> (Hcq)</td>
      <td>BFLOAT16</td>
      <td> (Hcq)</td>
    </tr>
    <tr>
      <td> rmsnorm_gamma_ckv </td>
      <td>BFLOAT16</td>
      <td> (Hckv)</td>
      <td>BFLOAT16</td>
      <td> (Hckv)</td>
      <td>BFLOAT16</td>
      <td> (Hckv)</td>
    </tr>
    <tr>
      <td> rope_sin </td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
    </tr>
    <tr>
      <td> rope_cos </td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
    </tr>
    <tr>
      <td> cache_index </td>
      <td>INT64</td>
      <td> · (B,S) <br> · (T)</td>
      <td>INT64</td>
      <td> · (B,S) <br> · (T)</td>
      <td>INT64</td>
      <td> · (B,S) <br> · (T)</td>
    </tr>
    <tr>
      <td> kv_cache </td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
      <td>INT8</td>
      <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
    </tr>
    <tr>
      <td> kr_cache </td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Dr)</td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Dr)</td>
      <td>INT8</td>
      <td> (BlockNum, BlockSize, Nkv, Dr)</td>
    </tr>
    <tr>
      <td> dequant_scale_x </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
    </tr>
    <tr>
      <td> dequant_scale_w_dq </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
    </tr>
    <tr>
      <td> dequant_scale_w_uq_qr </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, N*(D+Dr)) </td>
      <td>FLOAT</td>
      <td> (1, N*(D+Dr)) </td>
    </tr>
    <tr>
      <td> dequant_scale_w_dkv_kr </td>
      <td> 无需赋值 </td>
      <td> / </td>
      <td> 无需赋值 </td>
      <td> / </td>
      <td> 无需赋值 </td>
      <td> / </td>
    </tr>
    <tr>
      <td> dequant_scale_ckv </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, Hckv) </td>
    </tr>
    <tr>
      <td> dequant_scale_ckr </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, Dr) </td>
    </tr>
    <tr>
      <td> smooth_scales_cq </td>
      <td>无需赋值</td>
      <td>  </td>
      <td>FLOAT</td>
      <td> (1, Hcq) </td>
      <td>FLOAT</td>
      <td> (1, Hcq) </td>
    </tr>
    <tr>
      <td> query_out </td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
    </tr>
    <tr>
      <td> query_rope_out </td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
    </tr>
    <tr>
      <td> dequant_scale_q_nope </td>
      <td>None</td>
      <td>/</td>
      <td>None</td>
      <td>/</td>
      <td>None</td>
      <td>/</td>
    </tr>
    <tr>
      <td> query_norm </td>
      <td>BFLOAT16</td>
      <td> · (B, S, Hcq) <br> · (T, Hcq)</td>
      <td>INT8</td>
      <td> · (B, S, Hcq) <br> · (T, Hcq)</td>
      <td>INT8</td>
      <td> · (B, S, Hcq) <br> · (T, Hcq)</td>
    </tr>
    </tr>
     <tr>
      <td> dequant_scale_q_norm </td>
      <td>None</td>
      <td> / </td>
      <td>FLOAT32</td>
      <td> · (B * S) <br> · (T) </td>
      <td>FLOAT32</td>
      <td> · (B * S) <br> · (T) </td>
    </tr>
  </table>
  </div>


## 调用示例

- 详见 [test_npu_mla_prolog_v3.py](../examples/test_npu_mla_prolog_v3.py)