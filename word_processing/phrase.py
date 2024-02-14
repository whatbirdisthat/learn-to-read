class RandomPhrase:
    @property
    def mywords(self):
        #        with open("words-lists.yaml", "r") as wl_file:

        with open(self.words_list_filename, "r") as wl_file:
            import yaml
            words_lists = yaml.safe_load(wl_file)
        which_list = words_lists['list']
        import json
        # print(f'WORDS_LISTS: {json.dumps(words_lists, indent=4)}')
        return words_lists[which_list]

    def __init__(self, words_lists_filename="words-lists.yaml"):
        self.words_list_filename = words_lists_filename
        import random
        from word_processing.syllable import generate_syllables_console
        self.word = self.mywords[random.randint(0, len(self.mywords) - 1)]
        phrase_syllables = generate_syllables_console(self.word)
        print(f"[b]SYLLABLES: {phrase_syllables}[/]")

    def __str__(self):
        return self.word
