# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

from diffusers import StableDiffusionPipeline
import time

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

continueGeneration = True
imgs = 0
while(continueGeneration):
    imgs += 1
    print("Enter prompt:")
    prompt = str(input())
    pipe.nsfw_filter = False
    _ = pipe(prompt, num_inference_steps=1)
    image = pipe(prompt, height=512, width=512).images[0]
    file_name = "image " + str(imgs) + ".png"
    image.save(file_name)
    print("Continue?")
    if(input() == "y"):
        continueGeneration = True
    else:
        continueGeneration = False