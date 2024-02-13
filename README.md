# Learn to Read

I am writing this to help my son with his reading skills.
The idea is "a computerised flashcard reading trainer".

It will:

    1. Show the user a "flash card" phrase to read
    2. Provide the user with a means by which to record their attempt to read the phrase
    3. Compare the user's attempt to read the phrase with the original, using AI transcription
    4. [-] After 3 failed attempts, will "read out loud" the phrase, so the user can learn how to say it.

_Eventually_, the ~~look and feel~~ UX will improve, but for now it's a Gradio App.

# Prerequisites

This project doesn't really know how to do things without CUDA (I think).
It uses the highest level APIs I could find to do the things.

1. Dependencies:
```script
conda create -yn learn-to-read
conda activate learn-to-read

pip install -r requirements.txt

```

# Usage

```script
conda run -n learn-to-read --live-stream python app.py

```

---

_Some extra information._

| --- | --- |
| app.py |  Will display an app that does "real-time" transcription of input from the microphone, and|
|       |  compare it with the text of the "flash card" phrase.|

    * Some other implementation code, for reference etc.

    asr.app.py  Will show a web page with a "click the microphone button" type of transcription controls.
    microphone_interface.py    A toy web page that uses numpy to reverse a recording from the microphone.


