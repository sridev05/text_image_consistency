from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Load image
image = Image.open("data/images/test.jpg").convert("RGB")

# Generate caption
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=30)

caption = processor.decode(output[0], skip_special_tokens=True)

print("IMAGE DESCRIPTION:", caption)
