import os
import torch
import gc

import folder_paths
from .quickmt_third_party import Translator

models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")


MODEL_CACHE = None

class QuickMTRun:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":(["quickmt-en-zh", "quickmt-zh-en", "quickmt-en-fr", "quickmt-fr-en"],{"default": "quickmt-en-zh"}),
                "text": ("STRING", {"forceInput": True}),
                "beam_size": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "max_batch_size": ("INT", {"default": 32, "min": 1, "max": 100, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translations",)
    FUNCTION = "translate"
    CATEGORY = "🎤MW/MW-gemmax"
    def translate(self, model, text, beam_size, max_batch_size, temperature, top_k, top_p, unload_model):
        model_id = model_path + "/" + model

        import re
        texts = [i.strip() for i in re.split(r'\n\s*\n', text.strip()) if i.strip()]

        global MODEL_CACHE
        if MODEL_CACHE is None or self.model_name != model_id:
            self.model_name = model_id
            MODEL_CACHE = Translator(model_id, device=self.device)
            
        outputs = MODEL_CACHE(
            src=texts,
            beam_size=beam_size,
            max_batch_size=max_batch_size,
            sampling_temperature=temperature,
            sampling_topk=top_k,
            sampling_topp=top_p,
            # src_lang="en",
            # tgt_lang="zh",
        )

        outputs = "\n\n".join(outputs)

        if unload_model:
            MODEL_CACHE = None
            gc.collect()
            torch.cuda.empty_cache()

        return (outputs,)
