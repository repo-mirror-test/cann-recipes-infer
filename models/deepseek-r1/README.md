# DeepSeek-R1或Kimi-K2模型在NPU实现高性能推理

## 概述
DeepSeek-R1和Kimi-K2都是2025年开源的大语言模型，二者结构类似，代码可以复用。本样例基于[Deepseek开源代码](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/modeling_deepseek.py)进行迁移，并完成对应的优化适配。

## 支持的产品型号
<term>Atlas A3 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.3.RC1.alpha003`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha003)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Atlas-A3-cann-kernels_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`7.2.RC1.alpha003`，PyTorch版本为`2.6.0`。
   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/tags/v7.2.RC1.alpha003-pytorch2.6.0)下载`v7.2.RC1.alpha003-pytorch2.6.0`源码，参考[源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0005.html)。

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的python库，仅支持python 3.11
    cd cann-recipes-infer/models/deepseek-r1
    pip3 install -r requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段:
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `recipes_path`: 当前代码仓根目录，例如`/home/cann-recipes-infer`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。

   > 说明：HCCL相关配置，如：`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0001.html#ZH-CN_TOPIC_0000002449945377__section163522499503)并在`executor/scripts/function.sh`中自定义配置。

## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/models/origin/`。

Deepseek-R1与Kimi-K2的原始权重下载地址如下：
- [Deepseek-R1权重](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main)
- [Kimi-K2权重](https://huggingface.co/moonshotai/Kimi-K2-Instruct/tree/main)

## 权重转换
在各个节点上使用`weight_convert.sh` 脚本完成fp8到bfloat16/int8权重转换。

  >入参介绍：`input_fp8_hf_path`：原始fp8权重路径；`output_hf_path`：转换后输出的权重路径；`quant_mode`：量化模式

  权重转换拉起示例：
  ```
  # 转换为bfloat16权重，适用于DeepSeek-R1和Kimi-K2。
  bash utils/weight_convert.sh --input_fp8_hf_path /data/models/origin/DeepSeek-R1-FP8 --output_hf_path /data/models/origin/DeepSeek-R1-Bfloat16 --quant_mode bfloat16

  # 转换为W8A8C16权重，适用于DeepSeek-R1和Kimi-K2。
  bash utils/weight_convert.sh --input_fp8_hf_path /data/models/origin/DeepSeek-R1-FP8 --output_hf_path /data/models/origin/DeepSeek-R1-W8A8C16 --quant_mode w8a8c16

  # 转换为W8A8C8权重，仅适用于DeepSeek-R1。
  bash utils/weight_convert.sh --input_fp8_hf_path /data/models/origin/DeepSeek-R1-FP8 --output_hf_path /data/models/origin/DeepSeek-R1-W8A8C8 --quant_mode w8a8c8
  ```
  > 注意：仅DeepSeek-R1支持转W8A8C8权重。


## 推理执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   - 修改YAML文件中`model_path`参数。关于YAML文件中的更多配置说明可参见[YAML参数描述](./config/README.md)。

     在`models/deepseek-r1/config`目录下已提供了较优性能的YAML样例供您参考，您可以根据模型类型、集群规模以及量化类型选择对应的YAML文件，本文以`models/deepseek-r1/config/decode_r1_rank_16_16ep_a8w8.yaml`文件为例，修改其中的`model_path`参数，将其设置为[权重转换](#权重转换)阶段准备好的权重文件存储路径，例如`/data/models/origin/DeepSeek-R1-W8A8`。

   - 修改`models/deepseek-r1/infer.sh`脚本中`YAML_FILE_NAME`参数。

     将`YAML_FILE_NAME`设置为`config`文件夹下YAML文件名称，例如`decode_r1_rank_16_16ep_a8w8.yaml`。

2. 准备输入prompt。

   - 使用内置prompt。

     本样例已在`dataset/default_prompt.json`中内置了输入prompt，若您直接使用内置prompt，本步骤可直接跳过。

     当然，您也可以在`dataset/default_prompt.json`文件中自定义prompt输入。

   - 使用长序列prompt。

     本样例默认使用内置prompt，若您需要使用长序列prompt，需要执行以下操作：

     1. 修改YAML文件中的`dataset`参数，将其修改为`dataset: "LongBench"`，使用LongBench数据集作为长序列prompt。

     2. 若您的机器无法联网，需要您从[huggingface](http://huggingface.co/datasets/zai-org/LongBench/tree/main)手动下载数据集至`dataset/LongBench`目录下，`LongBench`文件夹需手工创建，目录中包含`LongBench.py`和`data`目录，并需要在`LongBench.py`中修改数据集加载路径；若您的机器可正常联网，样例执行过程中会自动在线读取LongBench数据集，您无需手工下载。

     > 说明：
     > - 使用LongBench数据集时，默认执行文本摘要任务，可在`cann-recipes-infer/executor/utils/data_utils.py`的`build_dataset_input`函数里修改默认的system prompt。
     > - 长序列请求执行中若出现`out of memory`问题，可参见附录中的[长序列请求out of memory问题处理](#long_bench_faq)。

3. 执行推理脚本。

   ```shell
   cd models/deepseek-r1
   bash infer.sh
   ```
   > 说明：如果是多机环境，需要在每个节点上执行。


## 优化点参考

- 本样例prefill阶段采用的详细优化点介绍可参见[基于Atlas A3集群的DeepSeek-R1模型prefill阶段推理性能优化实践](../../docs/models/deepseek-r1/deepseek_r1_prefill_optimization.md)。

- 本样例decode阶段采用的详细优化点介绍可参见[基于Atlas A3集群的DeepSeek-R1模型decode阶段推理性能优化](../../docs/models/deepseek-r1/deepseek_r1_decode_optimization.md)。

## 附录

### 常见问题处理

**长序列请求out of memory问题处理<a id="long_bench_faq"></a>**

长序列请求可能导致device内存out of memory，尤其是在perfill阶段:

- Attention的Softmax操作通常为float32计算，其内存大小为batch_size * num_heads * q_s * kv_s * (2Bytes + 4Bytes)。

- MoE的Routing分发，可能存在极端负载不均，导致个别卡上的grouped_matmul算子占用较大内存。

为了缓解这两处峰值带来的OOM问题，可分别采用以下方法：

  - 通过YAML文件中的`enable_pa`开关使能Flash Attention融合算子，算子内会切块计算Attention，避免了q_s * kv_s的峰值内存产生。

  - Prefill内存通常与batch_size大小成正比，当decode需要推理的global batch size过大时，prefill可能会由于OOM而无法在一轮推理中处理完所有的batch，因此我们可进行多次小batch串行推理，从而降低峰值内存。

    可通过YAML中的`enable_prefill_multi_cycle`开关使能，当前仅支持mini_batch的大小为1，即逐batch进行推理。

  - 为了缓解MoE负载不均带来的峰值内存，我们可进行Chunk MoE推理，即在MoE切Chunk串行推理，降低极端场景下的峰值内存，可通过YAML中的`moe_chunk_max_len`开关设置chunk的大小。当前该开关只针对prefill生效，开启后，由于MoE部分将串行计算各chunk，会对prefill的性能产生相应的影响。
