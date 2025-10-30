# cann-recipes-infer

## 🚀Latest News
- [2025/10] Wan2.2-I2V模型支持Ulysses序列并行、CFG并行、VAE并行，推理代码已开源。
- [2025/10] HunyuanVideo模型支持Ulysses序列并行、RingAttention序列并行、TeaCache加速，推理代码已开源。
- [2025/10] DeepSeek-V3.2-Exp模型支持W8A8C8量化，量化算法和推理代码已开源。
- [2025/10] Qwen3-MoE模型在昇腾Atlas A3系列上已支持推理部署。
- [2025/09] DeepSeek-V3.2-Exp模型在昇腾Atlas A3系列上已0day支持推理部署。


## 🎉概述
cann-recipes-infer仓库旨在针对LLM与多模态模型推理业务中的典型模型、加速算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地使用CANN平台进行模型推理。


## ✨样例列表
|实践|简介|
|-----|-----|
|[DeepSeek-V3.2-Exp](models/deepseek-v3.2-exp/README.md)|基于Transformers库，在Atlas A3环境中Prefill阶段采用了长序列亲和的CP并行策略，Decode阶段沿用大EP并行，同时从整网上设计了新的NPU融合Kernel和多流并行优化，实现了较高的吞吐推理性能。
|[Qwen3-MoE](models/qwen3_moe/README.md)|基于Transformers库，在Atlas A3环境中完成Qwen3-235B-A22B模型的适配优化，支持TP或EP部署。
|[HunyuanVideo](models/HunyuanVideo/README.md)|基于xDiT框架，在Atlas A2环境中采用了Ulysses序列并行和RingAttention序列并行测量，同时适配了TeaCache加速，实现了较高的吞吐推理性能。
|[Wan2.2-I2V](models/Wan2.2-I2V/README.md)|基于Transformers库，在Atlas A2环境中完成Wan2.2-I2V模型的适配优化。


## 📖目录结构说明
```
├── docs                                        # 文档目录
|  ├── models                                   # 模型文档目录
|  |  ├── deepseek-v3.2-exp                     # DeepSeek-V3.2-Exp相关文档
|  |  ├── qwen3_moe                             # Qwen3-MoE相关文档
|  |  ├── HunyuanVideo                          # HunyuanVideo相关文档
|  |  ├── Wan2.2-I2V                            # Wan2.2-I2V相关文档
|  |  └── ...
|  └── common                                   # 公共文档目录
├── accelerator                                 # 加速算法样例
├── executor                                    # ModelRunner等模型执行相关的类定义
|  ├── model_runner.py                          # ModelRunner类定义
│  └── ...
├── models                                      # 模型脚本目录
|  ├── deepseek-v3.2-exp                        # DeepSeek-V3.2-Exp的模型脚本及执行配置
|  ├── qwen3_moe                                # Qwen3-MoE的模型脚本及执行配置
|  ├── HunyuanVideo                             # HunyuanVideo的模型脚本及执行配置
|  ├── Wan2.2-I2V                               # Wan2.2-I2V的模型脚本及执行配置
│  └── ...
├── modules                                     # Linear等基础layer的类定义
│  └── linear.py                                # Linear类定义
│  └── ...
├── ops                                         # 算子目录
|  ├── ascendc                                  # ascendc算子
|  ├── pypto                                    # pypto算子
│  └── tilelang                                 # tilelang算子
└── CONTRIBUTION.md
└── README.md
└── ...
```

## 📝相关信息

- [贡献指南](./CONTRIBUTION.md)
- [许可证](./LICENSE)

    cann-recipes-infer仓涉及的模型，如模型目录下存在License的以该License为准。如模型目录下不存在License的，遵循CANN 2.0协议，对应协议文本可查阅[LICENSE](./LICENSE)
- [免责声明](./DISCLAIMER.md)
