from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load the pre-trained model
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Define a function to generate an image from a text prompt
def generate_image(prompt):
    # Generate the image
    with torch.no_grad():
        image = model(prompt).images[0]

    return image

# Test the function
prompt = "A beautiful sunset on a beach"
image = generate_image(prompt)
image.save("sunset_on_beach.png")
