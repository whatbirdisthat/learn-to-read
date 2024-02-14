# alpha_only_regex = regex.Regex(r"[^\p{L}]")
# regex_alpha_spaces = regex.Regex(r"[^\p{L}\s]")
# alpha_only_word = regex_alpha_spaces.sub("", result.text)

def strip_punctuation(text):
    import regex
    regex_alpha_spaces = regex.Regex(r"[^\p{L}\s]")
    alpha_only_word = regex_alpha_spaces.sub("", text)
    word_lower = alpha_only_word.lower()
    return word_lower
