# os.system("pip install git+https://github.com/openai/whisper.git")
import os
import torch
import whisper

from word_processing.cleaning import clean_text

model = None


class WhisperClient:
    def __init__(self, model_name="medium.en"):
        self.model_name = model_name
        self.device = 'cuda'
        if not torch.cuda.is_available():
            self.device = "cpu"

    @property
    def model(self):
        global model
        if model is None:
            model = whisper.load_model(self.model_name, device=self.device, in_memory=True)
        return model

    def asr(self, audio):
        if self.model_name.endswith('.en'):
            return self.en(audio)
        return self.multi_lingual(audio)

    def multi_lingual(self, audio):
        asr_model = self.model
        this_audio = whisper.load_audio(audio)
        this_audio = whisper.pad_or_trim(this_audio)
        mel = whisper.log_mel_spectrogram(this_audio).to(asr_model.device)
        _, probs = asr_model.detect_language(mel)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(asr_model, mel, options)
        clean_result = clean_text(result.text)
        return clean_result

    def en(self, audio):
        asr_model = self.model
        result = asr_model.transcribe(audio)
        clean_result = clean_text(result['text'])
        return clean_result


def asr(audio):
    whisper_client = WhisperClient()
    return whisper_client.asr(audio)

# def asr_multilingual(audio):
#     """
#     Detect the language of the audio and return the transcription.
#     This is better at understanding the Australian accent than
#     simply using the Whisper.transcribe() method.
#     :param audio: The audio file.
#     :return: The transcription.
#     """
#     asr_model = WhisperClient().model
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)
#     mel = whisper.log_mel_spectrogram(audio).to(asr_model.device)
#     _, probs = asr_model.detect_language(mel)
#     options = whisper.DecodingOptions(fp16=False)
#     result = whisper.decode(asr_model, mel, options)
#     clean_result = clean_text(result.text)
#     return clean_result
#
# def asr_transcribe(audio):
#     asr_model = WhisperClient().model
#     result = asr_model.transcribe(audio)
#     clean_result = clean_text(result['text'])
#     return clean_result
