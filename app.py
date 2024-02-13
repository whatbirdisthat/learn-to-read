from functools import cached_property

import gradio as gr
import numpy as np
import yaml
from transformers import pipeline
import time
from rich import print

pipe = pipeline("automatic-speech-recognition", model='openai/whisper-medium.en', device=0)
text2speech = pipeline("text-to-speech", model="suno/bark-small", device=0, model_kwargs={"pad_token_id": 10000})


class UI_Styles:
    @cached_property
    def app_styles(self):
        return '''
        #label_the_word { font-size: xx-large; color: gold; font-family: "Monospace" ; text-align: center; }
        '''

    def __str__(self):
        return UI_Styles.app_styles


def say(words):
    this_audio = text2speech(words)
    # synthesised_speech = (this_audio['audio'].numpy() * 32767).astype(np.int16)
    # synthesised_speech = (this_audio['audio'] * 32767).astype(np.int16)
    synthesised_speech = this_audio['audio'].T
    sample_rate = this_audio['sampling_rate']
    return sample_rate, synthesised_speech
    # return this_audio['sampling_rate'], this_audio['audio']


def transcribe(audio, state="", speakword=""):
    print(audio)
    time.sleep(2)
    text = pipe(audio)["text"].lower().strip()

    print(f"[blue]Transcribing:[/][green]'{text}'[/]")

    with open("skip-list.txt") as skip_list:
        hallucination_words = [each_line.rstrip() for each_line in skip_list.readlines()]

    # print("[orange]Hallucination Words:[/]")
    # print(hallucination_words)

    if text == "" or text in hallucination_words:
        print(f"[red i]'{text}' is a hallucination[/]")
    else:
        state += "\n" + text + " "
        if speakword in text:
            state += f"'{speakword}' is ✅ Correct! Well done.\n"
            speakword = f'{RandomWord()}'.lower()
        else:
            print(f"{speakword} ❌ Incorrect.")
            state += f"{speakword} ❌ Incorrect. Try again.\n"

    return state, state, speakword


class RandomWisdom:
    wise_sayings = [
        "📚📚📚 Learn to 🧑‍🔬 READ 📚📚📚",
        "Your 🧠 is a muscle. 💪 The more you use it, the stronger it gets. 🏋️‍♂️",
        "📖 Reading is a great way to exercise your brain. 🧠",
        "📖 Reading 📖  It helps you learn new things and understand the world around you. 🌍",
    ]

    def __init__(self):
        import random
        self.wisdom = self.wise_sayings[random.randint(0, len(self.wise_sayings) - 1)]

    def __str__(self):
        return self.wisdom


class RandomWord:
    @property
    def mywords(self):
        with open("words-lists.yaml", "r") as wl_file:
            words_lists = yaml.safe_load(wl_file)
        which_list = words_lists['list']
        import json
        print(f'WORDS_LISTS: {json.dumps(words_lists, indent=4)}')
        return words_lists[which_list]

    def __init__(self):
        import random
        self.word = self.mywords[random.randint(0, len(self.mywords) - 1)]

    def __str__(self):
        return self.word


with gr.Blocks(css=UI_Styles().app_styles) as demo:
    state = gr.State(value="")

    with gr.Row():
        with gr.Column():
            todays_wisdom = gr.Markdown(f"# {RandomWisdom()}")

    with gr.Row():
        the_random_word = RandomWord()
        the_word = gr.Label(f"{the_random_word}", elem_id='label_the_word')
        words_list = gr.Code("\n".join([w.rstrip() for w in the_random_word.mywords]))

    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone"], type="filepath", streaming=True)
        with gr.Column():
            textbox = gr.Textbox()

    with gr.Row():
        with gr.Column():
            audio_player = gr.Audio(type="numpy")
            # audio_player = gr.Audio(sources=[synthesised_speech], type="numpy", sample_rate=sample_rate)
        with gr.Column():
            some_words = gr.Markdown("# These are some words!")
            say_button = gr.Button("Say the word")
            say_button.click(say, inputs=[the_word], outputs=[audio_player])

    audio.stream(fn=transcribe, inputs=[audio, state, the_word], outputs=[textbox, state, the_word])

demo.launch(debug=True)
