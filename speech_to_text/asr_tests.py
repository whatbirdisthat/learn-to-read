import unittest
from functools import cached_property


class ASR_TestCase(unittest.TestCase):
    @cached_property
    def audio(self):
        return "../voicewav/richard-burton.wav"

    @cached_property
    def expected(self):
        return ' noises in a swound at length did cross an albatross thorough the fog it came as if it had been a christian soul we hailed it in gods name it ate the food it neer had ate and round and round it flew the ice did split'

    def test_openai_whisper(self):
        from asr_openai_whisper import asr
        actual = asr(self.audio)
        self.assertEqual(self.expected, actual)

    def test_whisper_transformers_pipeline(self):
        from asr_whisper_pipeline import asr
        actual = asr(self.audio)
        self.assertEqual(self.expected, actual)

    def test_t5_transformers_pipeline(self):
        from asr_t5_pipeline import asr
        actual = asr(self.audio)
        self.assertEqual(self.expected, actual)


if __name__ == '__main__':
    unittest.main()
