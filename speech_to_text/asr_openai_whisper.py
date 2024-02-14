# os.system("pip install git+https://github.com/openai/whisper.git")
import os

import regex
import whisper

# model = whisper.load_model("small")
model = None


class WhisperClient:
    def __init__(self, model_name="small"):
        self.model_name = model_name

    def model(self):
        global model
        if model is None:
            model = whisper.load_model(self.model_name)
        return model


def clean_text(text):
    from word_processing.cleaning import strip_punctuation
    alpha_only_words = strip_punctuation(text)

    if os.environ.get("DEBUG") == "1":
        print(f"cleaner: TEXT='{text}'")
        print(f"cleaner: REGEX='{alpha_only_words}'")

    return alpha_only_words


def asr(audio):
    asr_model = WhisperClient().model()

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(asr_model.device)

    _, probs = asr_model.detect_language(mel)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(asr_model, mel, options)

    clean_result = clean_text(result.text)
    return clean_result
