from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from collections import Counter

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load image
image = Image.open("data/images/test.jpg").convert("RGB")

# Prepare input
inputs = processor(image, return_tensors="pt")

# Generate multiple captions
num_captions = 3
captions = []

with torch.no_grad():
    for _ in range(num_captions):
        output = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)
        captions.append(caption.lower())  # lowercase to normalize

print("All Captions:", captions)

# Remove hallucinated/rare words by keeping common terms
# Split captions into words
words_list = [set(caption.split()) for caption in captions]

# Count word occurrences
word_counter = Counter()
for words in words_list:
    word_counter.update(words)

# Keep words that appear in at least 2 captions
common_words = [word for word, count in word_counter.items() if count >= 2]

# Create final caption
final_caption = " ".join(common_words)
print("REFINED IMAGE DESCRIPTION:", final_caption)
