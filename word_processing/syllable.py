from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize


def generate_syllables(phrase:str):
    # syllables = nltk.SyllableTokenizer().tokenize(word)
    # return syllables
    the_colours = 'blue black maroon green orange white purple red gold burgundy mahogany darkgreen'.split()
    the_phrase = word_tokenize(phrase)
    SSP = SyllableTokenizer()
    the_syllables = [s for s in [SSP.tokenize(token) for token in the_phrase]]
    colourised_syllables = [f"[{the_colours[i]}]{syllable}[/]" for i, syllable in enumerate(the_syllables)]
    return colourised_syllables
