[中文](README-CN.md)|[English](README.md)

# ComfyUI Translation Nodes. 

GemmaX2: Supports 28 languages. Arabic, Bengali, Czech, German, English, Spanish, Persian, French, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Burmese, Dutch, Polish, Portuguese, Russian, Thai, Tagalog, Turkish, Urdu, Vietnamese, Chinese.

**quickmt: Extremely fast, highly accurate translation**. More languages are being added, currently Chinese, English, and French models are available.

## 📣 Updates

[2025-05-29]⚒️: Added quickmt model node for fast and accurate translation. Added 4bit model for GemmaX2, which uses less VRAM and is faster. 

- https://huggingface.co/quickmt
- https://huggingface.co/Tonic/GemmaX2-28-2B-4bit

[2025-03-23]⚒️: Released version v1.0.0. 

## Usage

- Translates 1000 words in 0.5 seconds:

![](https://github.com/billwuhao/ComfyUI_gemmax/blob/main/images/2025-05-29_19-45-18.png)

![](https://github.com/billwuhao/ComfyUI_gemmax/blob/main/images/20250529195239.png)

- GemmaX2: 

![](https://github.com/billwuhao/ComfyUI_gemmax/blob/main/images/2025-03-23_07-12-01.png)

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_gemmax.git
cd ComfyUI_gemmax
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

Choose one of the following models and download it to the `ComfyUI/models/TTS` directory.

- [GemmaX2-28-2B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-2B-v0.1) 
- [GemmaX2-28-9B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1)
- [GemmaX2-28-2B-4bit](https://huggingface.co/Tonic/GemmaX2-28-2B-4bit)
- [quickmt](https://huggingface.co/quickmt) series.

## Acknowledgements

- [gemmax](https://github.com/xiaomi-research/gemmax)
- [quickmt](https://github.com/quickmt/quickmt)