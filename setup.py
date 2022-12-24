from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
import json


class ToxicDetector:
    def __init__(self, model_name="tomh/toxigen_roberta"):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir="models" # saves toxigen model in models
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir="tokenizers" # saves toxigen tokenizers in tokenizers
        )

    def classify(self, text: str) -> str:
        # tokenizes the input text and returns pytorch tensors
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        label = self.model.config.id2label[predicted_class_id]

        if label == "LABEL_0":
            label = "normal"
        else: # LABEL_1
            label = "hate"

        return label


class ResponseGenerator:
    def __init__(
        self,
        model_name="shaneweisz/DialoGPT-finetuned-gab-multiCONAN",
        decoding_config="decoding_config.json",
        verbose=False,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir="models"
        ) # saves counterspeech model in models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir="tokenizers"
        ) # saves counterspeech tokenizers in tokenizers

        # gets decoding config in dictionary form
        self.decoding_config = json.load(open(decoding_config))
        # verbose = multiple outputs
        self.verbose = verbose

        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, text: str) -> str:
        # tokenizes the input text and returns pytorch tensors
        input_ids = self.tokenizer.encode(
            text + self.tokenizer.eos_token, return_tensors="pt"
        )

        params_for_generate = self.decoding_config
        # generates with model
        reply_ids = self.model.generate(
            input_ids, **params_for_generate, pad_token_id=self.tokenizer.pad_token_id
        )

        # decodes output with tokenizer
        response = self.tokenizer.batch_decode(
            reply_ids[:, input_ids.shape[-1] :], skip_special_tokens=True
        )[0]

        return response


class DialoGPT:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir="models"
        ) # saves dialogpt model in models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", cache_dir="tokenizers"
        ) # saves dialogpt tokenizers in tokenizers

    def generate(self, text: str) -> str:
        # tokenizes the input text and returns pytorch tensors
        input_ids = self.tokenizer.encode(
            text + self.tokenizer.eos_token, return_tensors="pt"
        )

        # generates with model using added parameters
        reply_ids = self.model.generate(
            input_ids,
            max_length=1000,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.75,
            pad_token_id=self.tokenizer.eos_token_id,
            encoder_no_repeat_ngram_size=None,
        )

        # decodes output with tokenizer
        output = self.tokenizer.decode(
            reply_ids[:, input_ids.shape[-1] :][0], skip_special_tokens=True
        )

        return output
