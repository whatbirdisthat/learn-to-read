import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model = None
processor = None


class WhisperPipelineClient:
    def __init__(self, model_name="openai/whisper-large-v3"):
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    @property
    def model(self):
        global model
        if model is None:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                use_flash_attention_2=True
            )

            model.to(self.device)
        return model

    @property
    def processor(self):
        global processor
        if processor is None:
            processor = AutoProcessor.from_pretrained(self.model_name)
        return processor

    def asr(self, audio):
        asr_model = self.model
        asr_processor = self.processor

        pipe = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            tokenizer=asr_processor.tokenizer,
            feature_extractor=asr_processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
            generate_kwargs={"language": "english"}
        )

        audio_transcript = pipe(audio)
        return audio_transcript["text"]


def asr(audio):
    whisper_client = WhisperPipelineClient()
    return whisper_client.asr(audio)
