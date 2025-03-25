from transformers import AutoModelForCausalLM, AutoTokenizer
import folder_paths
import os
import torch

models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")

LANGUAGES = ["Arabic", "Bengali", "Czech", "German", "English", "Spanish", "Persian", "French", "Hebrew", "Hindi", 
             "Indonesian", "Italian", "Japanese", "Khmer", "Korean", "Lao", "Malay", "Burmese", "Dutch", "Polish", 
             "Portuguese", "Russian", "Thai", "Tagalog", "Turkish", "Urdu", "Vietnamese", "中文"]

class GemmaxRun:
    tokenizer = None
    model_cache = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":(["GemmaX2-28-2B-v0.1", "GemmaX2-28-9B-v0.1"],{"default": "GemmaX2-28-2B-v0.1"}),
                "source_language": (LANGUAGES, {"default": "English"}),
                "target_language": (LANGUAGES, {"default": "中文"}),
                "text": ("STRING",),
                "max_new_tokens": ("INT", {"default": 200, "min": 1,}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translations",)
    FUNCTION = "translate"
    CATEGORY = "🎤MW/MW-gemmax"
    def translate(self, model, source_language, target_language, text, max_new_tokens, unload_model):
        model_id = model_path + "/" + model
        if self.model_cache is None:
            model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer = tokenizer
            self.model_cache = model
        else:
            model = self.model_cache
            tokenizer = self.tokenizer

        text = "将文本从{}翻译成{}：\n\n{}:{}\n\n{}:".format(source_language, target_language, source_language, text, target_language)

        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        translations = tokenizer.decode(outputs[0], skip_special_tokens=True)

        translations = translations.split(f"\n\n{target_language}:")[-1].strip('"“”[] ')
        
        if unload_model:
            import gc
            self.tokenizer = None
            self.model_cache = None
            gc.collect()
            torch.cuda.empty_cache()

        return (translations,)


NODE_CLASS_MAPPINGS = {
    "GemmaxRun": GemmaxRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GemmaxRun": "Gemmax Run",
}