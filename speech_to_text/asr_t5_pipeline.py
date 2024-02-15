import torch
from rich import print
import scipy.io.wavfile
import numpy as np

# from transformers import pipeline
# import numpy as np

processor = None
model = None


class T5PipelineClient:
    def __init__(self, model_name="microsoft/speecht5_asr"):
        self.model_name = model_name
        self.device = 'cuda'

    @property
    def model(self):
        global model
        if model is None:
            from transformers import SpeechT5ForSpeechToText
            model = SpeechT5ForSpeechToText.from_pretrained(self.model_name)
        return model

    @property
    def processor(self):
        global processor
        if processor is None:
            from transformers import SpeechT5Processor
            processor = SpeechT5Processor.from_pretrained(self.model_name)
        return processor

    def asr(self, audio):

        asr_model = self.model
        asr_processor = self.processor

        sampling_rate, wav_bytes = scipy.io.wavfile.read(audio)
        if sampling_rate != 16000:
            wav_bytes = scipy.signal.resample(wav_bytes, int(len(wav_bytes) * 16000 / sampling_rate))
            sampling_rate = 16000
        # inputs = asr_processor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt")

        inputs = asr_processor(audio=wav_bytes, sampling_rate=sampling_rate, return_tensors="pt")

        if not torch.cuda.is_available():
            self.device = "cpu"
        # inputs.to(self.device)

        predicted_ids = asr_model.generate(**inputs, max_length=100)
        transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)

        print(f'[green]T5:[/] "{transcription[0]}"')
        the_words = transcription[0]
        return the_words


def asr(words):
    t5 = T5PipelineClient()
    return t5.asr(words)

    # ORIGINAL WORKING CODE
    # this_audio = text2speech(words)
    # transposed_audio = this_audio['audio'].T
    # sample_rate = this_audio['sampling_rate']
    # return sample_rate, transposed_audio
