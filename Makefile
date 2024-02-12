env:
	conda activate asr-tts

app:
	gradio app.py

asr:
	python unused_material/asr.py

mic:
	python unused_material/mic.py
