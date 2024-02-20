import os
import pathlib

import scipy
import torch.cuda
from TTS.api import TTS
import numpy as np

tt = None


class TTSClient:
    def __init__(self):
        self.cache_wav = "_ttsclient_cache.wav"
        self.output_wav = "_ttsclient_output.wav"
        self.default_voice = "voicewav/richard-burton.wav"
        self.voice_files_cache = [self.default_voice]
        # self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    @property
    def tts(self):
        global tt
        if tt is None:
            tt = TTS("tts_models/multilingual/multi-dataset/xtts_v2") #, gpu=torch.cuda.is_available())
            tt.to("cuda" if torch.cuda.is_available() else "cpu")
        return tt

    def update_voice_cache(self, more_audio):
        if not pathlib.Path(f"{more_audio}").exists():
            print(f"Voice file {more_audio} does not exist.")
            return

        sample_rate, new_audio = scipy.io.wavfile.read(more_audio)
        cache = np.array([])
        if pathlib.Path(self.cache_wav).exists():
            _, cache = scipy.io.wavfile.read(self.cache_wav)

        cache_t = cache.size / sample_rate
        print(f"Cache is {cache_t} seconds long.")
        cache = np.concatenate((cache, new_audio))
        scipy.io.wavfile.write(self.cache_wav, sample_rate, cache)
        print(f"Cache is {cache.size / sample_rate} seconds long.")
        # if cache_t < 10:
        #     cache = np.concatenate((cache, new_audio))
        #     scipy.io.wavfile.write(self.cache_wav, sample_rate, cache)
        # else:
        #     cache = new_audio
        #     print(f"Cache is {cache.size / sample_rate} seconds long.")

    def say(self, words, voicefile_path) -> tuple[int, np.ndarray]:
        if voicefile_path is None or not os.path.exists(voicefile_path):
            print(f"[red]Voice file {voicefile_path} does not exist.[/]")
            voicefile_path = self.default_voice
        if voicefile_path not in self.voice_files_cache:
            self.voice_files_cache.append(voicefile_path)
            self.update_voice_cache(voicefile_path)

        self.tts.tts_to_file(
            text=words,
            speaker_wav=voicefile_path,
            file_path=self.output_wav,
            language="en")  # load the file into an ndarray
        sample_rate, audio = scipy.io.wavfile.read(self.output_wav)

        # audio = np.load("output.wav")
        # sample_rate = audio['sampling_rate']
        return sample_rate, audio


t = None


def say(words: str, voicefile_path) -> tuple[int, np.ndarray]:
    """
    The T5 model sounds crisp, and the pipeline is faster than the others.
    It's an American voice however.
    :param words: a str containing the words to transcribe
    :return: sample rate, audio
    """
    global t
    if t is None:
        t = TTSClient()
    return t.say(words, voicefile_path)
