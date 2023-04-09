from parrot import Parrot
import torch
from datasets import MoviesDataset, FoodReviewsDataset
import re
from customized_one_line_summary import OneLineSummary
from transformers import pipeline
import dill
import os
from utils import remove_words_without_content, postprocess_phrases
import random


class ParrotTextualPCA(Parrot):
    def __init__(self, model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False, encoder_layer=1):
        super().__init__(model_tag=model_tag, use_gpu=use_gpu)
        self.avg_sentence = None
        self.avg_encoder_outputs = None

    def get_avg_embedding(self, input_phrases, use_gpu=False):
        if use_gpu:
            device = "cuda:0"
        else:
            device = "cpu"

        self.model = self.model.to(device)
        encoder = self.model.get_encoder()
        input_phrases = [re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase) for input_phrase in input_phrases]
        input_phrases = ["paraphrase: " + input_phrase for input_phrase in input_phrases]
        input_ids = \
            self.tokenizer.batch_encode_plus(input_phrases, padding=True, truncation=True, return_tensors='pt').data[
                "input_ids"]
        input_ids = input_ids.to(device)

        encoder_outputs = encoder(input_ids)
        changed_hidden_states = torch.mean(encoder_outputs.last_hidden_state, dim=0).unsqueeze(dim=0)
        changed_hidden_state = torch.mean(changed_hidden_states, dim=1).unsqueeze(dim=1)
        encoder_outputs.last_hidden_state = changed_hidden_state

        self.avg_encoder_outputs = encoder_outputs
        return encoder_outputs

    def get_avg_sentence(self, encoder_outputs=None, use_gpu=False):
        if use_gpu:
            device = "cuda:0"
        else:
            device = "cpu"

        self.model = self.model.to(device)

        preds = self.model.generate(
            None,
            do_sample=False,
            max_length=64,
            top_k=50,
            early_stopping=False,
            num_return_sequences=1,
            encoder_outputs=encoder_outputs,
            repetition_penalty=1.5,
            length_penalty=1.0,
        )

        paraphrases = set()

        for pred in preds:
            gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True, ).lower()
            gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
            paraphrases.add(gen_pp)

        self.avg_sentence = list(paraphrases)[0]
        return self.avg_sentence

    def generate_from_latent(self, encoder_outputs=None, use_gpu=False):
        if use_gpu:
            device = "cuda:0"
        else:
            device = "cpu"

        self.model = self.model.to(device)

        preds = self.model.generate(
            None,
            do_sample=True,
            max_length=12,
            top_k=256,
            top_p=0.9,
            early_stopping=False,
            num_return_sequences=256,
            encoder_outputs=encoder_outputs,
            repetition_penalty=1.5,
            length_penalty=2.0,
        )

        paraphrases = set()

        for pred in preds:
            gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True, ).lower()
            gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
            paraphrases.add(gen_pp)
        return list(paraphrases)

    def shorten_augment(self, input_phrase, use_gpu=False, diversity_ranker="levenshtein", do_diverse=False,
                        max_return_phrases=10, max_length=16, adequacy_threshold=0.90, fluency_threshold=0.90):
        if use_gpu:
            device = "cuda:0"
        else:
            device = "cpu"

        self.model = self.model.to(device)

        import re

        save_phrase = input_phrase
        if len(input_phrase) >= max_length:
            max_length += 32

        input_phrase = re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase)
        input_phrase = "paraphrase: " + input_phrase
        input_ids = self.tokenizer.encode(input_phrase, return_tensors='pt')
        input_ids = input_ids.to(device)
        preds = self.model.generate(
            input_ids,
            do_sample=False,
            max_length=max_length,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1,
            repetition_penalty=1.5,
            length_penalty=0,
        )

        paraphrases = set()

        for pred in preds:
            gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
            gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
            paraphrases.add(gen_pp)

        adequacy_filtered_phrases = self.adequacy_score.filter(input_phrase, paraphrases, adequacy_threshold, device)
        if len(adequacy_filtered_phrases) > 0:
            fluency_filtered_phrases = self.fluency_score.filter(adequacy_filtered_phrases, fluency_threshold, device)
            if len(fluency_filtered_phrases) > 0:
                diversity_scored_phrases = self.diversity_score.rank(input_phrase, fluency_filtered_phrases,
                                                                     diversity_ranker)
                para_phrases = []
                for para_phrase, diversity_score in diversity_scored_phrases.items():
                    para_phrases.append((para_phrase, diversity_score))
                para_phrases.sort(key=lambda x: x[1], reverse=True)
                return para_phrases
            else:
                return [(save_phrase, 0)]

    @staticmethod
    def remove_empty_phrases(phrases):
        return [phrase for phrase in phrases if phrase]

    @staticmethod
    def remove_mean_sentence_words(caption, avg_sentence):
        return " ".join([word for word in caption.split() if word not in avg_sentence.split()])

