from functools import cached_property

import gradio as gr

import yaml
import time
from rich import print

import argparse

from UI.styling import UI_Styles

# from speech_to_text.asr_pipeline import asr
from speech_to_text.asr_openai_whisper import asr
# from text_to_speech.tts_bark import say
from text_to_speech.tts_bark_voices import say

# from speech_to_text.asr_fake import asr
# from text_to_speech.tts_fake import say

from word_processing.phrase import RandomPhrase
from word_processing.wisdom import RandomWisdom

my_args = argparse.ArgumentParser(description="Read the arguments from the command line")
my_args.add_argument("--settings", help="The settings file", default="settings.yaml")
args = my_args.parse_args()

app_settings = yaml.safe_load(open(args.settings, "r"))
print(f"APP_SETTINGS: {app_settings}")


# app_settings['models']['text2speech'] = 'suno/bark-small'
# app_settings['words_lists'] = 'words-lists.yaml'

# the app will dream up a list of random words
# sentence_pieces = pipeline("fill-mask", model="bert-base-uncased", device=0)


def transcribe(audio, state="", speakword=""):
    print(audio)
    time.sleep(2)
    # text = asr(audio)["text"].lower().strip()
    asr_text = asr(audio)

    print(f"[b]Transcribing:[/][green]'{asr_text}'[/]")

    from word_processing.syllable import generate_syllables_plain, generate_syllables_console

    phrase_syllables = generate_syllables_console(asr_text)
    print(f"[b]SYLLABLES:[/][green]{phrase_syllables}[/]")
    label_syllables = f'{generate_syllables_plain(speakword)}'
    # html_syllables = generate_syllables_html(text)

    #    with open("skip-list.txt") as skip_list:
    with open(app_settings['lists']['skip-list']) as skip_list:
        hallucination_words = [each_line.rstrip() for each_line in skip_list.readlines()]

    # print("[orange]Hallucination Words:[/]")
    # print(hallucination_words)
    print(f"""[b]CHECKING: [/][blue]'{speakword}'[/] in [green]'{asr_text}'[/]""")

    if asr_text == "" or asr_text in hallucination_words:
        print(f"[red i]'{asr_text}' is a hallucination[/]")
    else:
        state += "\n" + asr_text + " "
        if speakword.lower() in asr_text:
            state += f"'{speakword}' is ✅ Correct! Well done.\n"
            speakword = f'{RandomPhrase(app_settings["lists"]["phrase-lists"])}'.lower()
            label_syllables = f'{generate_syllables_plain(speakword)}'
        else:
            print(f"{speakword} ❌ Incorrect.")
            state += f"{speakword} ❌ Incorrect. Try again.\n"

    return state, state, speakword, label_syllables


# IS_RECORDING = False
#
#
# def recording_stopped(audio, state="", speakword=""):
#     IS_RECORDING = False
#     print(f"Recording stopped: {audio}")
#     return audio, state, speakword
#
#
# def recording_started(audio, state="", speakword=""):
#     IS_RECORDING = True
#     print(f"Recording started: {audio}")
#     return audio, state, speakword


with gr.Blocks(css=UI_Styles().app_styles) as demo:
    state = gr.State(value="")

    with gr.Row():
        with gr.Column():
            todays_wisdom = gr.Markdown(f"# {RandomWisdom()}")
            wise_button = gr.Button("🧠 Get some wisdom 🧠 ")
        with gr.Column():
            wise_audio_player = gr.Audio(type="numpy", autoplay=True)
            wise_button.click(say, inputs=[todays_wisdom], outputs=[wise_audio_player])

    with gr.Row():
        with gr.Column():
            from word_processing.syllable import generate_syllables_html, generate_syllables_plain

            the_random_word = RandomPhrase(app_settings["lists"]["phrase-lists"])
            the_word = gr.Label(f"{the_random_word}", elem_id='label_the_word')
            the_syllables = gr.Label(generate_syllables_plain(f"{the_random_word}"))
        with gr.Column():
            words_list = gr.Code("\n".join([w.rstrip() for w in the_random_word.mywords]))

    with gr.Row():
        with gr.Column():
            # audio = gr.Audio(sources=["microphone"], type="filepath", streaming=True)
            audio = gr.Audio(sources=["microphone"], type="filepath", streaming=False)
        with gr.Column():
            textbox = gr.Textbox()

    with gr.Row():
        with gr.Column():
            audio_player = gr.Audio(type="numpy", autoplay=True)
            # audio_player = gr.Audio(sources=[synthesised_speech], type="numpy", sample_rate=sample_rate)
        with gr.Column():
            some_words = gr.Markdown("# These are some words!")
            say_button = gr.Button("Say the word")
            say_button.click(say, inputs=[the_word], outputs=[audio_player])

    # audio.stream(fn=transcribe, inputs=[audio, state, the_word], outputs=[textbox, state, the_word, the_syllables])

    # audio.start_recording(fn=recording_started)
    # audio.stop_recording(fn=recording_stopped)
    audio.stop_recording(fn=transcribe, inputs=[audio, state, the_word], outputs=[textbox, state, the_word, the_syllables])

    # audio.stop(transcribe, inputs=[audio, state, the_word], outputs=[textbox, state, the_word])

demo.launch(debug=True)
