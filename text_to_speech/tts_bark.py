from transformers import pipeline

text2speech = pipeline("text-to-speech", model="suno/bark-small", device=0, model_kwargs={"pad_token_id": 10000})


def say(words):
    this_audio = text2speech(words)
    transposed_audio = this_audio['audio'].T
    sample_rate = this_audio['sampling_rate']
    return sample_rate, transposed_audio
