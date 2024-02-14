import unittest


class WordProcessingTests(unittest.TestCase):
    def test_strip_punctuation(self):
        test_word = "A nice long stretch."
        expected = "a nice long stretch"

        from word_processing.cleaning import strip_punctuation
        actual = strip_punctuation(test_word)
        self.assertEqual(actual,expected)  # add assertion here

    def test_for_spaced_syllables(self):
        test_phrase = "compost"
        expected = "com post"

        from word_processing.syllable import generate_syllables_plain
        actual = generate_syllables_plain(test_phrase)
        self.assertEqual(actual,expected)

if __name__ == '__main__':
    unittest.main()
