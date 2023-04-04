from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from parrot import Parrot
import torch
import warnings
from typing import List, Tuple
from transformers import AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
from typing import List, Tuple
from abc import ABC, abstractmethod
from datasets import MoviesDataset


class OneLineSummary(ABC):
    """
    Class for one line summary using snrspeaks/t5-one-line-summary
    We augment this model to perform as textual PCA
    """

    def __init__(self, model_tag="snrspeaks/t5-one-line-summary", use_gpu=False, encoder_layer=1):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_tag)
        self.tokenizer = AutoTokenizer.from_pretrained(model_tag)
        self.device = "cuda" if use_gpu else "cpu"
        self.model = self.model.to(self.device)
        self.encode = self.model.get_encoder()


    def get_avg_encode_outputs(self, sentences):
        sentences = ["summarize: " + sentence for sentence in sentences]
        tokenized = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").data["input_ids"]
        encoder_outputs = self.encode(tokenized)
        avg_last_hidden_state = torch.mean(encoder_outputs.last_hidden_state, dim=1).unsqueeze(dim=1)
        avg_sentences_lats_hidden_state = torch.mean(avg_last_hidden_state, dim=0).unsqueeze(dim=0)
        encoder_outputs.last_hidden_state = avg_sentences_lats_hidden_state
        return encoder_outputs

    def generate_sentences(self, input_ids=None, encoder_outputs=None):
        generated_ids = self.model.generate(input_ids=input_ids, num_beams=1, max_length=50, repetition_penalty=2.0,
                                            length_penalty=1, early_stopping=True, num_return_sequences=1,
                                            encoder_outputs=encoder_outputs)
        preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                 generated_ids]
        return preds


if __name__ == "__main__":
    one_line_summary = OneLineSummary()
    movie_dataset = MoviesDataset(genres=["comedy"])
    movies_synopsis = [movie_dataset[i]["Synopsis"] for i in range(0, 100)]
    embedding = one_line_summary.get_avg_encode_outputs(sentences=movies_synopsis)
    preds = one_line_summary.generate_sentences(encoder_outputs=embedding)

    print(preds)
