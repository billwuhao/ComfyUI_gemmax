from pathlib import Path
from typing import Optional

import huggingface_hub
from fire import Fire
from huggingface_hub import HfApi


def hf_list():
    """List quickmt models available on Huggingface

    Returns:
        None
    """
    api = HfApi()
    models = api.list_models(author="quickmt")
    return [i.id for i in models]


def hf_upload(
    repo_id: str,
    input_dir: str,
    joint_vocab: bool = False
):
    """Uploads a quickmt model to the Hugging Face Hub.
    # adaptions from https://github.com/michaelfeil/hf-hub-ctranslate2/blob/main/hf_hub_ctranslate2/util/utils.py

    Args:
      repo_name: repo name on HF Hub e.g.  "quickmt/quickmt-zh-en"
      input_dir: Local directory containing model files.

    Returns:
      None
    """

    api = HfApi()

    input_path = Path(input_dir)
    eole_model_path = Path(input_dir)/"eole-model"
    if joint_vocab:
        for f in (
            "README.md",
            "config.json",
            "model.bin",
            "shared_vocabulary.json",
            "joint.spm.model",
            "joint.eole.vocab",
            "joint.spm.vocab",
            "eole-config.yaml"
        ):
            assert Path(input_path / f).is_file(), f"Cannot upload - must include {f}"
 
    else:
        for f in (
            "README.md",
            "config.json",
            "model.bin",
            "source_vocabulary.json",
            "src.spm.model",
            "target_vocabulary.json",
            "tgt.spm.model",
            "eole-config.yaml"
        ):
            assert Path(input_path / f).is_file(), f"Cannot upload - must include {f}"

    for f in (
        "config.json",
        "vocab.json",
        "model.00.safetensors"
    ):
        assert Path(eole_model_path / f).is_file(), f"Cannot upload - must include {f}"

    

    api.upload_folder(
        folder_path=input_dir,
        repo_id=repo_id,
        repo_type="model",
    )


def hf_download(
    model_name: str,
    output_dir: Optional[str] = None,
):
    """Downloads a quickmt model to the Hugging Face Hub.
    # adaptions from https://github.com/michaelfeil/hf-hub-ctranslate2/blob/main/hf_hub_ctranslate2/util/utils.py

    Args:
      model_name: repo name on HF Hub e.g.  "quickmt/quickmt-zh-en"
      output_dir: Directory where the model should be saved. Optional. Defaults to model name.

    Returns:
      None

    Raises:
      ValueError: if the model size is invalid.
    """
    if not output_dir:
        output_dir = model_name.split("/")[-1]

    allow_patterns = [
        "README.md",
        "config.json",
        "model.bin",
        "source_vocabulary.json",
        "target_vocabulary.json",
        "shared_vocabulary.json",
        "src.spm.model",
        "tgt.spm.model",
        "joint.spm.model",
        "eole-config.yaml"
    ]

    return huggingface_hub.snapshot_download(
        model_name, allow_patterns=allow_patterns, local_dir_use_symlinks=False, local_dir=output_dir
    )


def download():
    Fire(hf_download)


def upload():
    Fire(hf_upload)


def list():
    Fire(hf_list)


if __name__ == "__main__":
    Fire({"download": hf_download, "upload": hf_upload, "list": hf_list})
