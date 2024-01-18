from PIL import Image

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

cat_url = "cat.jpg"
dog_url = "dog.jpg"

image_dog = Image.open(dog_url)
image_cat = Image.open(cat_url)

arr = ["a photo of a cat", "a photo of a dog"]

inputs_dog = processor(text=arr, images=image_dog, return_tensors="pt", padding=True)
inputs_cat = processor(text=arr, images=image_cat, return_tensors="pt", padding=True)

outputs_dog = model(**inputs_dog)
logits_per_image_dog = outputs_dog.logits_per_image
probs_dog = logits_per_image_dog.softmax(dim=1)

outputs_cat = model(**inputs_cat)
logits_per_image_cat = outputs_cat.logits_per_image
probs_cat = logits_per_image_cat.softmax(dim=1)

probs_dog = probs_dog.squeeze()
probs_cat = probs_cat.squeeze()

print(f"Image {dog_url} is detected as a dog at {probs_dog[1]} %")
print(f"Image {cat_url} is detected as a cat at {probs_cat[0]} %")