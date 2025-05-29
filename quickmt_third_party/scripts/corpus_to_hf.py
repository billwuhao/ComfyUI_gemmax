import datasets
from datasets import Dataset
from fire import Fire


def data_generator(src_in: str, src_lang: str, tgt_in: str, tgt_lang):
    """Read in txt files and generate dicts"""
    with open(tgt_in, "rt") as t_in:
        with open(src_in, "rt") as s_in:
            for s, t in zip(s_in, t_in):
                yield {src_lang: s.strip(), tgt_lang: t.strip(), "sco": 1.0}


def corpus_to_hf(dataset_key, src_in: str, tgt_in: str, src_lang: str, tgt_lang: str):
    """Upload mt corpus to huggingface"""
    dataset = Dataset.from_generator(
        data_generator,
        gen_kwargs={
            "src_in": src_in,
            "tgt_in": tgt_in,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
        },
    )

    dataset.push_to_hub(dataset_key)


def main():
    Fire(corpus_to_hf)


if __name__ == "__main__":
    main()
