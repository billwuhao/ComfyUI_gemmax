from .gemmaxnode import GemmaxRun
from .quickmtnode import QuickMTRun


NODE_CLASS_MAPPINGS = {
    "GemmaxRun": GemmaxRun,
    "QuickMTRun": QuickMTRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GemmaxRun": "Gemmax Translate",
    "QuickMTRun": "QuickMT Translate",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
