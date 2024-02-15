import os


def strip_punctuation(text):
    import regex
    regex_alpha_spaces = regex.Regex(r"[^\p{L}\s]")
    alpha_only_word = regex_alpha_spaces.sub("", text)
    word_lower = alpha_only_word.lower()
    return word_lower


def clean_text(text):
    alpha_only_words = strip_punctuation(text)

    if os.environ.get("DEBUG") == "1":
        print(f"cleaner: TEXT='{text}'")
        print(f"cleaner: REGEX='{alpha_only_words}'")

    return alpha_only_words
