from fire import Fire

from ..translator import Translator


def translate_file(
    model_path: str,
    src_file: str,
    output_file: str,
    beam_size: int = 5,
    intra_threads: int = 1,
    inter_threads: int = 6,
    compute_type="auto",
    device: str = "cpu",
    max_batch_size: int = 32,
    max_decoding_length: int = 256,
):
    t = Translator(
        model_path=model_path,
        device=device,
        compute_type=compute_type,
        inter_threads=inter_threads,
        intra_threads=intra_threads,
    )

    t.translate_file(
        src_file,
        output_file,
        beam_size=beam_size,
        max_batch_size=max_batch_size,
        max_decoding_length=max_decoding_length,
    )


def main():
    Fire(translate_file)


if __name__ == "__main__":
    main()
