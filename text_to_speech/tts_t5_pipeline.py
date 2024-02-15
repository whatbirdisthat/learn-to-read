import torch
from transformers import pipeline
from datasets import load_dataset
import numpy as np

# text2speech = pipeline("text-to-speech", model="suno/bark-small", device=0, model_kwargs={"pad_token_id": 10000})
text2speech = None
embeddings_dataset = None


class T5PipelineClient:
    @property
    def pipeline(self):
        global text2speech
        if text2speech is None:
            text2speech = pipeline(
                "text-to-speech",
                model="microsoft/speecht5_tts"
            )
        return text2speech

    @property
    def embeddings_dataset(self):
        global embeddings_dataset
        if embeddings_dataset is None:
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        return embeddings_dataset

    def say(self, words, voice_preset="v2/en_speaker_0"):
        tts_pipeline = self.pipeline
        # this_audio = tts_pipeline(words)

        # embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        this_embeddings_dataset = self.embeddings_dataset
        speaker_embedding = torch.tensor(this_embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        this_audio = tts_pipeline(words, forward_params={"speaker_embeddings": speaker_embedding})

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

        # audio_array = transposed_audio.cpu().numpy().squeeze()
        # audio_array = convert_to_int16(audio_array)
        transposed_audio = convert_to_int16(transposed_audio)
        audio_array = transposed_audio.squeeze()
        # audio_array = convert_to_int16(audio_array)
        return sample_rate, audio_array


def say(words: str) -> tuple[int, np.ndarray]:
    """
    The T5 model sounds crisp, and the pipeline is faster than the others.
    It's an American voice however.
    :param words: a str containing the words to transcribe
    :return: sample rate, audio
    """
    t5 = T5PipelineClient()
    return t5.say(words)
