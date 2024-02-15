import random

from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize


def generate_syllables_console(phrase: str, mode: str = 'Console'):
    # syllables = nltk.SyllableTokenizer().tokenize(word)
    # return syllables
    the_colours = 'green orange white purple red green orange white purple red green orange white purple red'.split()
    the_phrase = word_tokenize(phrase)
    SSP = SyllableTokenizer()
    the_syllables = [s for s in [SSP.tokenize(token) for token in the_phrase]]
    colourised_syllables = [f"[{the_colours[len(the_colours) % 2]}]{syllable}[/]" for i, syllable in
                            enumerate(the_syllables)]
    return colourised_syllables


def generate_syllables_plain(phrase: str) -> str:
    """generates a plain text representation of the syllables of the phrase
    each syllable is an alternating colour
    """
    the_phrase = word_tokenize(phrase)
    SSP = SyllableTokenizer()
    the_syllables = [s for s in [SSP.tokenize(token) for token in the_phrase]]
    syllables_spaced = " ".join([f"{' '.join([s for s in syllable])}" for i, syllable in enumerate(the_syllables)])

    return syllables_spaced


def generate_syllables_html(phrase: str) -> str:
    """generates an HTML formatted representation of the syllables of the phrase
    each syllable is an alternating colour
    """

    the_colours = 'green orange white purple gold aliceblue darkgreen'.split()

    # the_colours = 'green orange white purple pink'.split()
    def colourise(syllable):
        colour = random.Random().randint(0, len(the_colours) - 1)
        return " ".join([f"<span style='color:{the_colours[colour]}'>{s}</span>" for s in syllable])

    phrase_words = word_tokenize(phrase)
    ssp = SyllableTokenizer()
    the_syllables = [colourise(s) for s in [ssp.tokenize(token) for token in phrase_words]]
    colourised_syllables = " ".join(
        [f"<span class='a_word'>{each_word}</span>" for _, each_word in enumerate(the_syllables)])

    return colourised_syllables
