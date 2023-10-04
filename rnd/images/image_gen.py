# https://huggingface.co/docs/diffusers/optimization/mps
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "A space station in orbit around the moon, earth in the background. Photorealistic. Milkyway"

# First-time "warmup" pass if PyTorch version is 1.13 (see explanation above)
_ = pipe(prompt, num_inference_steps=1)


image = pipe(prompt).images[0]
image.save("astronaut_rides_horse.png")
