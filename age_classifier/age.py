# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-classification", model="nateraw/vit-age-classifier")
pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")      Copy # Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")