import torch
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)

# ---------- LOAD MODELS ----------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# ---------- LOAD IMAGE ----------
image_path = "data/images/test.jpg"
image = Image.open(image_path).convert("RGB")

user_text = "a man with a beard wearing a white shirt"

# ---------- STEP 1: AUTO-CAPTION ----------
inputs = blip_processor(image, return_tensors="pt")
with torch.no_grad():
    out = blip_model.generate(**inputs, max_new_tokens=30)

caption = blip_processor.decode(out[0], skip_special_tokens=True)

# ---------- STEP 2: CLIP SIMILARITY ----------
clip_inputs = clip_processor(
    text=[user_text, caption],
    images=image,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = clip_model(**clip_inputs)

img_emb = outputs.image_embeds
txt_emb = outputs.text_embeds

img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

score_user = (img_emb @ txt_emb[0].T).item()
score_caption = (img_emb @ txt_emb[1].T).item()

final_score = (score_user + score_caption) / 2

# ---------- STEP 3: VERDICT ----------
if final_score >= 0.35:
    verdict = "CONSISTENT"
elif final_score >= 0.20:
    verdict = "PARTIALLY CONSISTENT"
else:
    verdict = "INCONSISTENT"

# ---------- OUTPUT ----------
print("User Text:", user_text)
print("Image Caption:", caption)
print("Final Score:", round(final_score, 3))
print("Verdict:", verdict)
