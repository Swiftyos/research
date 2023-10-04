import whisper

model = whisper.load_model("base")
result = model.transcribe("tts_example.wav")

print(result["text"])
