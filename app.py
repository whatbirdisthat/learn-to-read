from functools import cached_property

import gradio as gr

import yaml
import time
from rich import print

import argparse

from UI.styling import UI_Styles

from speech_to_text.asr_openai_whisper import asr
# from text_to_speech.tts_t5_pipeline import say
from text_to_speech.tts_xtts import say
from word_processing.cleaning import clean_text

from word_processing.phrase import RandomPhrase
from word_processing.wisdom import RandomWisdom

my_args = argparse.ArgumentParser(description="Read the arguments from the command line")
my_args.add_argument("--settings", help="The settings file", default="settings.yaml")
args = my_args.parse_args()

app_settings = yaml.safe_load(open(args.settings, "r"))
print(f"APP_SETTINGS: {app_settings}")

# the app will dream up a list of random words
# sentence_pieces = pipeline("fill-mask", model="bert-base-uncased", device=0)


def transcribe(audio_filename, app_state, speakword, wordlist_control):
    print(audio_filename)
    time.sleep(2)
    # text = asr(audio_filename)["text"].lower().strip()
    raw_asr = asr(audio_filename)
    asr_text = clean_text(raw_asr)

    print(f"[b]Transcribing:[/][green]'{asr_text}'[/]")

    from word_processing.syllable import generate_syllables_plain, generate_syllables_console

    phrase_syllables = generate_syllables_console(asr_text)
    print(f"[b]SYLLABLES:[/][green]{phrase_syllables}[/]")
    label_syllables = f'{generate_syllables_plain(speakword)}'

    with open(app_settings['lists']['skip-list']) as skip_list:
        hallucination_words = [each_line.rstrip() for each_line in skip_list.readlines()]

    # print("[orange]Hallucination Words:[/]")
    # print(hallucination_words)
    print(f"""[b]CHECKING: [/][blue]'{speakword}'[/] in [green]'{asr_text}'[/]""")
    message = "Try again... ❌"
    if asr_text == "" or asr_text in hallucination_words:
        print(f"[red i]'{asr_text}' is a hallucination[/]")
    else:
        app_state += "\n" + asr_text + " "
        if speakword.lower() in asr_text:
            message = f"'{asr_text}' is ✅ Correct! Well done.\n"
            randomword = RandomPhrase(app_settings["lists"]["phrase-lists"])
            speakword = f'{randomword}'.lower()
            label_syllables = f'{generate_syllables_plain(speakword)}'
            wordlist_control = print_words_list(randomword)
        else:
            print(f"{speakword} ❌ Incorrect.")
            message = f"{asr_text} ❌ Incorrect. Try again: '{speakword}'\n"

    return app_state, message, speakword, label_syllables, wordlist_control


def print_words_list(the_random_word):
    def is_now_word(a_word):
        a_word = a_word.rstrip()
        if a_word == the_random_word.word:
            return f"✅ {a_word}"
        else:
            return a_word

    return "\n* ".join([is_now_word(w) for w in the_random_word.mywords])


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
        with gr.Column():
            the_syllables = gr.Label(generate_syllables_plain(f"{the_random_word}"))

    with gr.Row():
        with gr.Column():
            # audio = gr.Audio(sources=["microphone"], type="filepath", streaming=True)
            learner_audio = gr.Audio(sources=["microphone"], type="filepath", streaming=False)
        with gr.Column():
            message_text = gr.Label()

    with gr.Row():
        with gr.Column():
            audio_player = gr.Audio(type="numpy", autoplay=True)
        with gr.Column():
            some_words = gr.Markdown("# These are some words!")
            say_button = gr.Button("Say the word")
            say_button.click(say, inputs=[the_word, learner_audio], outputs=[audio_player])

    with gr.Row():
        with gr.Column():
            words_list = gr.Markdown(print_words_list(the_random_word))

    # audio.stream(fn=transcribe, inputs=[audio, state, the_word], outputs=[textbox, state, the_word, the_syllables])
    learner_audio.stop_recording(fn=transcribe, inputs=[learner_audio, state, the_word, words_list],
                         outputs=[state, message_text, the_word, the_syllables, words_list])

demo.launch(debug=True)
