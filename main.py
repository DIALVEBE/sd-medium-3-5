import torch
from diffusers import StableDiffusion3Pipeline

print("Cargando modelo...")

modelo = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float32
)

modelo = modelo.to("cpu")

prompt = "A futuristic university proffesor teaching artificial intelligence inside a modern high-tech classroom, standing next to a large transparenteholographic display showing neural networks, data visualizations and machine learnign diagrams. Univerisity students are sitting at their desks, some using laptops, realistic facila expressions, modern architecture, large windows overlooking a fusturistic city, ceinematic natural lighting, volumetric light, realistic materials, detailed clothing, proffesional photography, shallow depth of field, 35 mm lens, highly detailed, photorealistic, sharp focus."

negative_prompt = """
blurry, low quality, low resolution, distorted, deformed,
bad anatomy, extra fingers, missing fingers, extra limbs,
duplicate people, poorly drawn hands, poorly drawn face,
cropped, out of frame, text, watermark, logo, oversaturated
"""

print("Generando imagen...")

imagen = modelo(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    guidance_scale=4.5,
    height=1024,
    width=1024
).images[0]

imagen.save("imagen_sd35.png")

print("Imagen guardada como imagen_sd35.png")
