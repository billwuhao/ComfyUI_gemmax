from time import time

import datasets
from fire import Fire
from sacrebleu import BLEU, CHRF, TER

from ..translator import Translator

bleu = BLEU()
chrf = CHRF()
ter = TER()


def eval(
    model_path: str,
    src_lang: str,
    tgt_lang: str,
    output_file: str,
    beam_size: int = 5,
    intra_threads: int = 1,
    inter_threads: int = 6,
    compute_type="auto",
    device: str = "auto",
    max_batch_size: int = 32,
    max_decoding_length: int = 512,
):
    t = Translator(
        model_path=model_path,
        device=device,
        compute_type=compute_type,
        inter_threads=inter_threads,
        intra_threads=intra_threads,
    )

    print("Loading flores-devtest data")
    try:
        flores = datasets.load_dataset(
            "facebook/flores",
            f"{src_lang}-{tgt_lang}",  # trust_remote_code=True
        )
    except:
        flores = datasets.load_dataset(
            "facebook/flores",
            f"{tgt_lang}-{src_lang}",  # trust_remote_code=True
        )

    src = []
    ref = []
    for i in flores["devtest"]:
        src.append(i[f"sentence_{src_lang}"])
        ref.append(i[f"sentence_{tgt_lang}"])

    print("Translating")
    mt = t(src, max_batch_size=max_batch_size, max_decoding_length=max_decoding_length, beam_size=beam_size)

    # Write results to file, for COMET scoring
    if output_file:
        with open(output_file, "wt") as myfile:
            myfile.write("".join([i.replace("\n", "\t") + "\n" for i in mt]))

    print("Source sample: ", src[:10])
    print("Reference sample: ", ref[:10])
    print("Translation sample: ", mt[:10])

    print(bleu.corpus_score(mt, [ref]))
    print(chrf.corpus_score(mt, [ref]))
    print(ter.corpus_score(mt, [ref]))


def main():
    Fire(eval)


if __name__ == "__main__":
    main()
