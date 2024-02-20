env:
	conda activate asr-tts

app:
	gradio app.py

asr:
	python unused_material/asr.py

mic:
	python unused_material/mic.py

clear-cache:
	rm -f _ttsclient_cache.wav
	rm -f _ttsclient_output.wav


