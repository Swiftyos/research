from transformers import pipeline

generator = pipeline(
    task="automatic-speech-recognition", model="microsoft/speecht5_asr"
)
generator.to("mps")

from datasets import load_dataset

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
)
dataset = dataset.sort("id")
example = dataset[40]
transcription = generator(example["audio"]["array"])
print(transcription)
