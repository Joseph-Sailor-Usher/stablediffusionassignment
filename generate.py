"""
This script will generate images via Stable Diffusion. It runs on Apple Silicon (M1/M2).

Based on: https://huggingface.co/docs/diffusers/optimization/mps
"""

from diffusers import StableDiffusionPipeline
import time

imgs = 0
continueGeneration = True
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

prompt = "gopro filming an underwater sunken mall"

# Toggle the NSFW filter
#
# The filter is intended to prevent the model from generating images that are
# inappropriate for the general public. However, it is not perfect and can
# sometimes block images that are not NSFW. If you wish to disable the filter,
# set the following variable to False.
pipe.nsfw_filter = True
 
# Enable sliced attention computation.
#
# When this option is enabled, the attention module will split the input tensor in slices, 
# to compute attention in several steps. This is useful to save some memory in exchange for 
# a small speed decrease.
#
# Per Hugging Face, recommended if your computer has < 64 GB of RAM.
pipe.enable_attention_slicing()

# First-time "warmup" pass
#
# This is necessary to get the model to run on Apple Silicon (M1/M2). There is a
# bug in the MPS implementation of the model that causes it to crash on the first
# pass. This is a workaround to get the model to run on Apple Silicon.
#
# It takes about 30 seconds to run.
_ = pipe(prompt, num_inference_steps=1)

imageNum = 1
infNum = 10
guidNum = 7
while(continueGeneration):
    print("Change prompt?")
    if(input() == "y"):
        print("Enter prompt:")
        prompt = str(input())
    print("Amount:")
    imgNum = int(input())
    print("Passes:")
    infNum = int(input())
    print("Guidance multiplier:")
    guidNum = int(input())

    # Results match those from the CPU device after the warmup pass.
    images = pipe(prompt, num_inference_steps=infNum, num_images_per_prompt=imageNum, guidance_scale=guidNum).images

    # loop through images
    for image in images:
        imgs += 1
        epoch_time = int(time.time())

        file_name = str(imgs) + str(epoch_time) + "_" + prompt + ".png"
        image.save(file_name)


    print("Continue?")
    if(input() == "y"):
        continueGeneration = True
    else:
        continueGeneration = False
