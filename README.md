# Learn to Read

I am writing this to help my son with his reading skills.
The idea is "a computerised flashcard reading trainer".

It will:

    1. Show the user a "flash card" phrase to read
    2. Provide the user with a means by which to record their attempt to read the phrase
    3. Compare the user's attempt to read the phrase with the original, using AI transcription
    4. Read the phrase "out loud", so the user can learn how to say it.

_Eventually_, the ~~look and feel~~ UI/UX will improve, but for now it's a Gradio App.

# Prerequisites

This project doesn't really know how to do things without CUDA (I think). I haven't tried it with anything other
than a NVIDIA GPU.
It uses the highest level APIs (transformers pipelines etc) to do the things.

I'm not sure if python 3.12 is supported by all the libraries I'm using, so I'm using 3.11.

1. Dependencies:
```script
conda create -yn learn-to-read python=3.11
conda activate learn-to-read

pip install -r requirements.txt

```

# Usage

```script
conda run -n learn-to-read --live-stream python app.py

```

---

_Some extra information._

| Script                  | Description                                                                                                                                    |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| app.py                  | Will display an app that does "real-time" transcription of input from the microphone, and compare it with the text of the "flash card" phrase. |
| gradio.rt.asr.py        | Sample code from [the gradio.app documentation](https://www.gradio.app/guides/real-time-speech-recognition)                                    |
| microphone_interface.py | A toy web page that uses numpy to reverse a recording from the microphone.                                                                     |
| asr.app.py              | Will show a web page with a "click the microphone button" type of transcription controls.                                                      |
 ---
