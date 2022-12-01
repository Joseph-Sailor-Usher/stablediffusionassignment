# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

from diffusers import StableDiffusionPipeline
import time

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

prompt = "a photo of an astronaut riding a horse on mars"

# First-time "warmup" pass (see explanation above)

_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
image = pipe(prompt).images[0]

for image in images:
    epoch_time = int(time.time())
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:100]

    file_name = str(epoch_time) + "_" + prompt + ".png"
    image.save(file_name)