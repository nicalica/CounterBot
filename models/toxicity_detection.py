import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ToxicDetector:
    def __init__(self, model_name="tomh/toxigen_roberta"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def classify(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        label = self.model.config.id2label[predicted_class_id]

        if label == "LABEL_0":
            label = "normal"
        else:
            label = "hate"

        return label
