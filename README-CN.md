[中文](README-CN.md)|[English](README.md)

# ComfyUI 翻译节点. 

GemmaX2: 支持 28 种语言. Arabic, Bengali, Czech, German, English, Spanish, Persian, French, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Burmese, Dutch, Polish, Portuguese, Russian, Thai, Tagalog, Turkish, Urdu, Vietnamese, 中文.

**quickmt: 极快速度, 超准确翻译**. 正在新增多种语言, 目前有中文, 英语, 法语模型可用.

## 📣 更新

[2025-05-29]⚒️: 增加 quickmt 模型节点, 又快又准的翻译. 增加 GemmaX2 的 4bit 模型, 显存占用更小, 速度更快. 

- https://huggingface.co/quickmt
- https://huggingface.co/Tonic/GemmaX2-28-2B-4bit

[2025-03-23]⚒️: 发布版本 v1.0.0. 

## 使用

- 0.5 秒翻译 1000 字:

![](https://github.com/billwuhao/ComfyUI_gemmax/blob/main/images/2025-05-29_19-45-18.png)

![](https://github.com/billwuhao/ComfyUI_gemmax/blob/main/images/20250529195239.png)

- GemmaX2:

![](https://github.com/billwuhao/ComfyUI_gemmax/blob/main/images/2025-03-23_07-12-01.png)

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_gemmax.git
cd ComfyUI_gemmax
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

将下列模型任选其一, 下载放到 `ComfyUI/models/TTS` 目录下.

- [GemmaX2-28-2B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-2B-v0.1) 
- [GemmaX2-28-9B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1)
- [GemmaX2-28-2B-4bit](https://huggingface.co/Tonic/GemmaX2-28-2B-4bit)
- [quickmt](https://huggingface.co/quickmt) 系列.

## 鸣谢

- [gemmax](https://github.com/xiaomi-research/gemmax)
- [quickmt](https://github.com/quickmt/quickmt)