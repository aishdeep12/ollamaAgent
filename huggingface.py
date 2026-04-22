import os
from pyclbr import Class
from huggingface_hub import InferenceClient

class HuggingFaceClient:

    def generate_image(self, prompt: str, image_title: str):

        client = InferenceClient(
            provider="wavespeed",
            api_key=os.getenv("HF_API_KEY"),
        )

        # output is a PIL.Image object
        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-dev",
        )

        image.save(f"output_{image_title}.png")