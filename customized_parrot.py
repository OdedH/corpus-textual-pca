from parrot import Parrot
import torch
import warnings
from typing import List, Tuple
from transformers import AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
import re


class CustomizedParrot(Parrot):
    def __init__(self, model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False, encoder_layer=1):
        super().__init__(model_tag=model_tag, use_gpu=use_gpu)
        # self.end_of_model = AutoModelForSeq2SeqLM.from_pretrained(model_tag, use_auth_token=False,
        #                                                           num_hidden_layers=encoder_layer)
        # self.encoder_layer = encoder_layer

    def get_avg_embedding(self, input_phrases, use_gpu=False):
        if use_gpu:
            device = "cuda:0"
        else:
            device = "cpu"

        self.model = self.model.to(device)
        encoder = self.model.get_encoder()
        input_phrases = [re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase) for input_phrase in input_phrases]
        input_phrases = ["paraphrase: " + input_phrase for input_phrase in input_phrases]
        input_ids = self.tokenizer.batch_encode_plus(input_phrases, return_tensors='pt').data["input_ids"]
        input_ids = input_ids.to(device)

        encoder_outputs = encoder(input_ids)
        changed_hidden_state = torch.mean(encoder_outputs.last_hidden_state, dim=1).unsqueeze(dim=1)
        encoder_outputs.last_hidden_state = changed_hidden_state

        return encoder_outputs

    def generate_from_latent(self, input_phrase=None, encoder_outputs=None, use_gpu=False, max_length=64):
        if use_gpu:
            device = "cuda:0"
        else:
            device = "cpu"

        self.model = self.model.to(device)

        preds = self.model.generate(
            input_phrase,
            do_sample=False,
            max_length=max_length,
            top_k=50,
            top_p=0.99,
            early_stopping=True,
            num_return_sequences=1,
            encoder_outputs=encoder_outputs,
            repetition_penalty=1.5,
        )

        paraphrases = set()

        for pred in preds:
            gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
            gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
            paraphrases.add(gen_pp)

        return list(paraphrases)


if __name__ == "__main__":
    parrot = CustomizedParrot()
    embeds = parrot.get_avg_embedding(["I want to go to the beach"])
    print(parrot.generate_from_latent(encoder_outputs=embeds))
    # parrot.generate_from_latent("I am a student.")
    # parrot.rephrase(["Can you recommed some upscale restaurants in Newyork?"])
