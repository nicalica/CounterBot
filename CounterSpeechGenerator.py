from typing import List, Tuple
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2Generator:
    def __init__(self, model_name='testmodel'):
        self.TARGET_TO_HS_TOK = {'DISABLED': '<|DISABLED_HS|>',
                            'JEWS': '<|JEWS_HS|>',
                            'LGBT+': '<|LGBT+_HS|>',
                            'MIGRANTS': '<|MIGRANTS_HS|>',
                            'MUSLIMS': '<|MUSLIMS_HS|>',
                            'POC': '<|POC_HS|>',
                            'WOMEN': '<|WOMEN_HS|>',
                            'other': '<|other_HS|>'}
        self.TARGET_TO_CS_TOK = {'DISABLED': '<|DISABLED_CS|>',
                            'JEWS': '<|JEWS_CS|>',
                            'LGBT+': '<|LGBT+_CS|>',
                            'MIGRANTS': '<|MIGRANTS_CS|>',
                            'MUSLIMS': '<|MUSLIMS_CS|>',
                            'POC': '<|POC_CS|>',
                            'WOMEN': '<|WOMEN_CS|>',
                            'other': '<|other_CS|>'}

        # dynamic load model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name + '_tokenizer')
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_counterspeech(self, texts: List[str], labels: List[str]) -> List[str]:

        # don't generate special tokens
        bad_words = list(self.TARGET_TO_HS_TOK.values()) + list(self.TARGET_TO_CS_TOK.values())
        bad_word_ids = [self.tokenizer(bad_word).input_ids[0] for bad_word in bad_words]

        tokenized_prompts = []
        responses = []

        for text, label in zip(texts, labels):

            hs_tok = self.TARGET_TO_HS_TOK[label]
            cs_tok = self.TARGET_TO_CS_TOK[label]

            prompt = hs_tok + text + cs_tok

            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").input_ids #.cuda()
            tokenized_prompts.append(tokenized_prompt)

        # max_len = max([len(p) for p in tokenized_prompts])

        for prompt in tokenized_prompts:
            # take best response
            output = self.model.generate(prompt, do_sample=True, top_k=30, bad_word_ids=bad_word_ids,
                                    max_length=20, top_p=0.95, temperature=1.9, num_return_sequences=1)[0]
            decoded_output = self.tokenizer.decode(output, skip_special_tokens=True)

            _, response = decoded_output.split(cs_tok)
            responses.append(response)

        return responses
