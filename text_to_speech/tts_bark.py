from transformers import pipeline
import numpy as np

text2speech = None


class BarkPipelineClient:
    @property
    def pipeline(self):
        global text2speech
        if text2speech is None:
            text2speech = pipeline(
                "text-to-speech",
                model="suno/bark-small",
                device=0,
                model_kwargs={"pad_token_id": 10000})
        return text2speech

    def say(self, words):
        this_audio = self.pipeline(words)
        transposed_audio = this_audio['audio'].T
        sample_rate = this_audio['sampling_rate']

        def convert_to_int16(data):
            if data.dtype == np.float32:
                data = data / np.abs(data).max()
                data = data * 32767
                data = data.astype(np.int16)
            else:
                print(f"Data type: {data.dtype}")
            return data

        audio_array = transposed_audio.cpu().numpy().squeeze()
        audio_array = convert_to_int16(audio_array)

        return sample_rate, audio_array
        # return audio_array


def say(words: str) -> tuple[int, np.ndarray]:
    """
    The Bark models sound really good, and the pipeline is easy to use.
    The only problem is that the models keep saying "um" and "ahh" all the time.
    :param words: the words to say
    :return: sample rate, audio
    """
    bark = BarkPipelineClient()
    return bark.say(words)
    # ORIGINAL WORKING CODE
    # this_audio = text2speech(words)
    # transposed_audio = this_audio['audio'].T
    # sample_rate = this_audio['sampling_rate']
    # return sample_rate, transposed_audio
