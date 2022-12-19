from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoGPT:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, text: str) -> str:
        input_ids = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors="pt")

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

        output = self.tokenizer.decode(reply_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        return output
