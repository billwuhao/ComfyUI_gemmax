import re
import sys

import fasttext
import nltk
from fire import Fire
from nltk.corpus import words
from sacremoses import MosesDetokenizer

try:
    word_list = words.words()

except:
    nltk.download("words")
    word_list = words.words()

eng_words = set(words.words())

# build a table mapping all non-printable characters to None
NOPRINT_TRANS_TABLE = {i: None for i in range(0, sys.maxunicode + 1) if not chr(i).isprintable()}

detok = MosesDetokenizer(lang="en")


def fasttext_lang_match(
    s,
    t,
    slang: str,
    tlang: str,
    ft,
    s_min_score: float = 0.5,
    t_min_score: float = 0.5,
):
    """Ensure correct source and target language via fasttext langid model"""
    if s_min_score == 0:
        s_lang = ft.predict(s, k=1)
        if s_lang[0][0].replace("__label__", "") != slang or s_lang[1][0] < s_min_score:
            return False
    if t_min_score > 0:
        t_lang = ft.predict(t, k=1)
        if t_lang[0][0].replace("__label__", "") != tlang or t_lang[1][0] < t_min_score:
            return False
    return True


def english_text_match(s_clean: str, t_clean: str, src_lang: str, tgt_lang: str):
    """Ensure English side has sufficient words and alpha chars

    Somewhat similar to https://github.com/mozilla/translations/blob/main/pipeline/clean/tools/clean_parallel.py#L73
    """
    if src_lang == "en":
        str_in = s_clean
    elif tgt_lang == "en":
        str_in = t_clean
    else:
        return True

    # 60% of words must contain letters
    toks = str_in.split()
    num_alpha = sum([1 if re.match(r"[a-z]", t, re.IGNORECASE) else 0 for t in toks])

    if num_alpha / float(len(toks)) < 0.4:
        return False

    # Must have at least 50% letters
    char_alpha = len(re.findall(r"[a-z]", str_in, re.IGNORECASE))
    if char_alpha / float(len(str_in.replace(" ", ""))) < 0.5:
        return False

    # Must not have any segments with "word" greater than 25 chars
    # Will filter out long URLs...
    if any([len(i) > 25 for i in toks]):
        return False

    # Must have no more than 25% non-English words
    num_words = sum([1 if t.lower() in eng_words else 0 for t in toks])
    if num_words / float(len(toks)) < 0.25:
        return False

    # Must have at least 2 English "words"
    if num_words < 3:
        return False

    return True


def char_length_match(
    s_clean: str,
    t_clean: str,
    min_char_length: int,
    max_char_length: int,
    length_ratio: float,
):
    """Ensure src/tgt within char length bounds and remove if identical src/tgt"""
    slen = len(s_clean)
    tlen = len(t_clean)
    if slen == 0 or tlen == 0:
        return False

    len_ratio = slen / tlen
    if (slen < min_char_length) or (tlen < min_char_length):
        return False
    if (slen > max_char_length) or (tlen > max_char_length):
        return False
    if (len_ratio < 1 / length_ratio) or (len_ratio > length_ratio):
        return False
    if s_clean == t_clean:
        return False

    return True


def clean_input(
    s,
    t,
    src_lang: str,
    tgt_lang: str,
    ft,
    src_min_langid_score: float = 0.5,
    tgt_min_langid_score: float = 0.5,
    length_ratio: int = 4,
    min_char_length: int = 3,
    max_char_length: int = 2000,
):
    """Parallel data filter and clean"""
    # for s, t in tqdm(zip(svec, tvec)):
    # Remove non-printable chars
    s_printable = s.translate(NOPRINT_TRANS_TABLE)
    t_printable = t.translate(NOPRINT_TRANS_TABLE)

    # Remove non-utf8 chars
    s_clean = s_printable.encode("utf-8", errors="ignore").decode("utf-8").replace("\t", " ").replace("￨", "|")
    t_clean = t_printable.encode("utf-8", errors="ignore").decode("utf-8").replace("\t", " ").replace("￨", "|")

    # Min/max char length and copy filter
    if char_length_match(s_clean, t_clean, min_char_length, max_char_length, length_ratio):
        # English only word/alphabet filter
        if english_text_match(s_clean, t_clean, src_lang, tgt_lang):
            # Langid filter
            if fasttext_lang_match(
                s_clean,
                t_clean,
                src_lang,
                tgt_lang,
                ft,
                src_min_langid_score,
                tgt_min_langid_score,
            ):
                # Detokenize target with sacremoses
                return "\t".join([s_clean, detok.detokenize([t_clean])]) + "\n"


def clean(
    src_lang: str,
    tgt_lang: str,
    src_min_langid_score: float = 0.5,
    tgt_min_langid_score: float = 0.5,
    length_ratio: float = 4,
    min_char_length: int = 3,
    max_char_length: int = 2000,
    ft_model_path: str = "../lid.176.bin",
):
    """Remove non-printable characters and filter out if char length ratio > `length_ratio`"""
    # https://stackoverflow.com/questions/66353366/cant-suppress-fasttext-warning-load-model-does-not-return
    fasttext.FastText.eprint = lambda x: None
    ft = fasttext.load_model(ft_model_path)

    for line in sys.stdin:
        fields = line.strip().split("\t")
        if len(fields) != 2:
            continue

        src_in = fields[-2].strip()
        tgt_in = fields[-1].strip()

        try:
            cleaned_input = clean_input(
                src_in,
                tgt_in,
                src_lang,
                tgt_lang,
                ft,
                src_min_langid_score,
                tgt_min_langid_score,
                length_ratio,
                min_char_length,
                max_char_length,
            )
            if cleaned_input:
                sys.stdout.write(cleaned_input)
        except:
            pass


def main():
    Fire(clean)


if __name__ == "__main__":
    main()
