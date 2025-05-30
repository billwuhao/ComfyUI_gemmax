from transformers import AutoModelForCausalLM, AutoTokenizer
import folder_paths
import os
import torch
import gc


models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")

LANGUAGES = ["Arabic", "Bengali", "Czech", "German", "English", "Spanish", "Persian", "French", "Hebrew", "Hindi", 
             "Indonesian", "Italian", "Japanese", "Khmer", "Korean", "Lao", "Malay", "Burmese", "Dutch", "Polish", 
             "Portuguese", "Russian", "Thai", "Tagalog", "Turkish", "Urdu", "Vietnamese", "中文"]

MODEL_CACHE = None
TOKENIZER = None
class GemmaxRun:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":(["GemmaX2-28-2B-v0.1", "GemmaX2-28-9B-v0.1", "GemmaX2-28-2B-4bit", "GemmaX2-28-2B-8bit"],{"default": "GemmaX2-28-2B-4bit"}),
                "source_language": (LANGUAGES, {"default": "English"}),
                "target_language": (LANGUAGES, {"default": "中文"}),
                "text": ("STRING", {"forceInput": True}),
                "max_new_tokens": ("INT", {"default": 200, "min": 1,}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translations",)
    FUNCTION = "translate"
    CATEGORY = "🎤MW/MW-gemmax"
    def translate(self, model, source_language, target_language, text, max_new_tokens, unload_model):
        model_id = model_path + "/" + model
        global MODEL_CACHE, TOKENIZER
        if MODEL_CACHE is None or self.model_name != model_id:
            self.model_name = model_id
            MODEL_CACHE = AutoModelForCausalLM.from_pretrained(model_id).eval().to(self.device)
            TOKENIZER = AutoTokenizer.from_pretrained(model_id)

        text = "将文本从{}翻译成{}：\n\n{}:{}\n\n{}:".format(source_language, target_language, source_language, text, target_language)

        inputs = TOKENIZER(text, return_tensors="pt").to(self.device)
        outputs = MODEL_CACHE.generate(**inputs, max_new_tokens=max_new_tokens)
        translations = TOKENIZER.decode(outputs[0], skip_special_tokens=True)

        translations = translations.split(f"\n\n{target_language}:")[-1].strip('"“”[] ')
        
        if unload_model:
            TOKENIZER = None
            MODEL_CACHE = None
            gc.collect()
            torch.cuda.empty_cache()

        return (translations,)
