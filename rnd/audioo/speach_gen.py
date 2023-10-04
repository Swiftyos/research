from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torch

# https://huggingface.co/blog/speecht5

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
inputs = processor(
    text="Can you tell me how to grow my buisness online using twitter",
    return_tensors="pt",
)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

from transformers import SpeechT5HifiGan

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

import soundfile as sf

sf.write("tts_example.wav", speech.numpy(), samplerate=16000)
