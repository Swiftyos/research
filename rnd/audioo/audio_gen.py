# https://huggingface.co/blog/audioldm2
# https://huggingface.co/blog/run-musicgen-as-an-api

# import scipy
# import torch
# from diffusers import AudioLDM2Pipeline

# repo_id = "cvssp/audioldm2"
# pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
# pipe = pipe.to("mps")

# # define the prompts
# prompt = "The sound of a hammer hitting a wooden surface."
# negative_prompt = "Low quality."

# # set the seed for generator
# generator = torch.Generator("mps").manual_seed(0)

# # run the generation
# audio = pipe(
#     prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=200,
#     audio_length_in_s=10.0,
#     num_waveforms_per_prompt=3,
#     generator=generator,
# ).audios

# # save the best audio sample (index 0) as a .wav file
# scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])


from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
model.to("mps")

inputs = processor(
    text=[
        "80s pop track with bassy drums and synth",
        "90s rock song with loud guitars and heavy drums",
    ],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)
scipy.io.wavfile.write("musicgen_large.wav", rate=16000, data=audio_values[0])
