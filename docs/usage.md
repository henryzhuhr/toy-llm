---
lastUpdated: true
editLink: true
footer: true
outline: deep
---


# 项目使用文档

## 项目介绍

本项目是一个大模型的集成，计划完成包括对话交流、指令分解、文本实体抽取、文本生成等功能，目前项目处于早期阶段。

## 项目进展

- 「2025.02.07」 支持 DeepSeek-R1 和 Qwen2.5
- 「2024.06.09」 支持基于 2024.6.7 开源的 [Qwen2](https://qwenlm.github.io/blog/qwen2/)([Code](https://github.com/QwenLM/Qwen2)) 模型进行指令微调的任务，目前仅完成了对话交流功能
- 「2024.04.28」 基于千文大模型(Qwen1.5)的模型导出和模型推理测试，并且完成了上下文对话的测试，能记录用户的对话历史
- 「2024.04.26」 项目启动

## 模型仓库

| 模型|  模型大小  | 模型地址 |
|---|---|---|
| Qwen2-0.5B | 0.5B | [Hugging Face](https://huggingface.co/Qwen/Qwen2-0.5B) / [ModelScope](https://www.modelscope.cn/qwen/Qwen2-0.5B) |
| Qwen2-0.5B-Instruct | 0.5B | [Hugging Face](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) / [ModelScope](https://www.modelscope.cn/qwen/Qwen2-0.5B-Instruct) |
| Qwen2-1.5B | 1.5B | [Hugging Face](https://huggingface.co/Qwen/Qwen2-1.5B) / [ModelScope](https://www.modelscope.cn/models/qwen/Qwen2-1.5B) |
| Qwen2-1.5B-Instruct | 1.5B | [Hugging Face](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) / [ModelScope](https://www.modelscope.cn/qwen/Qwen2-1.5B-Instruct) |
| Qwen2-7B | 7B | [Hugging Face](https://huggingface.co/Qwen/Qwen2-7B) / [ModelScope](https://www.modelscope.cn/models/qwen/Qwen2-7B) |
| Qwen2-7B-Instruct | 7B | [Hugging Face](https://huggingface.co/Qwen/Qwen2-7B-Instruct) / [ModelScope](https://www.modelscope.cn/qwen/Qwen2-7B-Instruct) |

<!-- 
## 环境要求
本项目在以下环境中测试通过：

| 系统          | CPU       | GPU      |
| ------------- | --------- | -------- |
| Ubuntu 22.04  | i9-13900K | RTX 4090 |
| Sonoma 12.0.1 | M1 Pro    | M1 Pro   | -->

## 环境配置

### 获取源码

获取项目源码

::: code-group

```shell [HTTP]
git clone https://github.com/henryzhuhr/toyllm
cd toyllm
```

```shell [SSH]
git clone git@github.com:henryzhuhr/toyllm.git
cd toyllm
```

:::

### 文档本地启动

该项目的文档可以本地启动，运行

```shell
pnpm docs:dev
```

### 创建环境

确保安装了 conda ，如果没有安装，请从 [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) 下载，或者快速安装
  
```shell
# linux x64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

创建环境，会在当前目录下 `.env/toyllm` 创建环境，并安装依赖

```shell
conda create -n toyllm python -y
conda activate toyllm
```

## 获取模型

### 下载模型

### 链接模型

为了方便使用，可以链接模型到 `.cache/models/` 目录下

```shell
mkdir -p .cache/models
ln -s /path/to/model .
# 例如 模型下载到 ~/data/DeepSeek-R1-Distill-Qwen-1.5B
ln -s ~/data/DeepSeek-R1-Distill-Qwen-1.5B .cache/models/
```

## 项目使用说明

### 导出模型

请到 [Hugging Face](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f) / [ModelScope](https://www.modelscope.cn/home) 下载 Qwen2 模型，以 [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct) 为例，下载模型到 `downloads` 目录下

```shell
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-7B downloads/Qwen/Qwen2-7B
```

> 模型遵循 [Apache-2.0](https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/LICENSE) 协议，请遵守协议使用

导出模型，默认导出至 `weights` 目录下

```shell
python export.py \
    --model_id Qwen/Qwen2-7B \
    --weight_dir downloads/Qwen/Qwen2-7B \
    --quan_type int8
```

### 模型推理

对话交流模型功能，以及预设了部分对话，可以直接运行方面测试模型效果，运行

```shell
python infer-chat.py \
    --model_id Qwen/Qwen2-7B \
    --model_path weights/Qwen/Qwen2-7B-IR-int8 \
    --quan_type int8 \
    --max_sequence_length 512
```

推理结果中可以看到，该模型可以记录用户的对话历史，并根据上下文进行对话

## 参考

- 🚀 通义千问 [QwenLM/Qwen2](https://github.com/QwenLM/Qwen2)

## License

本项目遵循 [The MIT License (MIT)](https://mit-license.org) 协议
