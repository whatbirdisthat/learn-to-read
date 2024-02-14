import torch
from transformers import AutoProcessor, BarkModel

# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = AutoProcessor.from_pretrained("suno/bark")
# model = BarkModel.from_pretrained("suno/bark").to(device)
model = None
processor = None


class BarkClient:
    def __init__(self, model_name="suno/bark"):
        self.model_name = model_name
        self.device = 'cuda'

    @property
    def model(self):
        global model
        if model is None:
            # model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)
            model = BarkModel.from_pretrained(self.model_name).to(self.device)
        return model

    @property
    def processor(self):
        global processor
        if processor is None:
            processor = AutoProcessor.from_pretrained("suno/bark")
        return processor

    def say(self, words, voice_preset="v2/en_speaker_6"):
        if not torch.cuda.is_available():
            self.device = "cpu"
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = self.model
        tts_processor = self.processor

        inputs = tts_processor(words, voice_preset=voice_preset)
        inputs.to(self.device)

        audio_array = tts_model.generate(
            **inputs,
            pad_token_id=tts_processor.tokenizer.pad_token_id,
            semantic_max_new_tokens=len(words) * 24
        )

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = tts_model.generation_config.sample_rate
        return sample_rate, audio_array


def say(words: str, voice_preset="v2/en_speaker_6"):
    bark = BarkClient()
    return bark.say(words, voice_preset=voice_preset)

    # inputs = processor(words, voice_preset=voice_preset)
    # inputs.to(device)
    #
    # audio_array = model.generate(
    #     **inputs,
    #     pad_token_id=processor.tokenizer.pad_token_id,
    #     semantic_max_new_tokens=len(words) * 24
    # )
    #
    # audio_array = audio_array.cpu().numpy().squeeze()
    # sample_rate = model.generation_config.sample_rate
    # return sample_rate, audio_array
    #
# voice_preset = "v2/en_speaker_6"
#
# inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)
#
# audio_array = model.generate(**inputs)
# audio_array = audio_array.cpu().numpy().squeeze()
# sample_rate = model.generation_config.sample_rate
#
