from transformers import pipeline

asr_pipeline = pipeline("automatic-speech-recognition", model='openai/whisper-medium.en', device=0)


def asr(audio):
    result = asr_pipeline(audio)
    the_text = result['text'].lower().strip()
    return the_text
