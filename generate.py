# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

from diffusers import StableDiffusionPipeline
import time

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

continueGeneration = True
imgs = 0
pipe.nsfw_filter = False
while(continueGeneration):
    print("Enter prompt:")
    prompt = str(input())

    imgs += 1
    _ = pipe(prompt, num_inference_steps=1)
    image = pipe(prompt).images[0]
    file_name = "image " + str(imgs) + ".png"
    image.save(file_name)

    imgs += 1
    _ = pipe(prompt, num_inference_steps=2)
    image = pipe(prompt).images[0]
    file_name2 = "image " + str(imgs) + ".png"
    image.save(file_name2)

    print("Continue?")
    if(input() == "y"):
        continueGeneration = True
    else:
        continueGeneration = False
