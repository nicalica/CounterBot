from transformers import AutoModelForCausalLM, AutoTokenizer
import json


class ResponseGenerator:
    def __init__(self, pretrained_model_name="shaneweisz/DialoGPT-finetuned-gab-multiCONAN",
                 decoding_config="decoding_config.json", verbose=False):
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.decoding_config = json.load(open(decoding_config))
        self.verbose = verbose

        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, text: str) -> str:
        input_ids = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors="pt")

        params_for_generate = self.decoding_config
        reply_ids = self.model.generate(
            input_ids,
            **params_for_generate,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response = self.tokenizer.batch_decode(reply_ids[:, input_ids.shape[-1]:], skip_special_tokens=True)[0]

        return response
