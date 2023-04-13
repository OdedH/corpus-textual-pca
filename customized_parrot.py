from parrot import Parrot
import torch
from datasets import MoviesDataset, FoodReviewsDataset
import re
from customized_one_line_summary import OneLineSummary
from transformers import pipeline
import dill
import os
import random
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from torch.utils.data import DataLoader
from typing import List


class ParrotTextualPCA(Parrot):
    def __init__(self, model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False, texts_encoder=None):
        super().__init__(model_tag=model_tag, use_gpu=use_gpu)
        self.mean_phrase = None
        self.avg_encoder_outputs = None
        self.prev_principal_phrases = []
        if texts_encoder is None:
            self.texts_encoder = self.model.get_encoder()
        else:
            self.texts_encoder = texts_encoder
        self.texts_projections = None
        self.mean_embedding = None
        self.device = "cuda" if use_gpu else "cpu"

    def get_avg_embedding(self, input_phrases, device="cuda"):
        self.device = device
        self.model = self.model.to(self.device)
        encoder = self.model.get_encoder()
        input_phrases = [re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase) for input_phrase in input_phrases]
        input_phrases = ["paraphrase: " + input_phrase for input_phrase in input_phrases]
        input_ids = \
            self.tokenizer.batch_encode_plus(input_phrases, padding=True, truncation=True, return_tensors='pt').data[
                "input_ids"]
        input_ids = input_ids.to(self.device)

        encoder_outputs = encoder(input_ids)
        changed_hidden_states = torch.mean(encoder_outputs.last_hidden_state, dim=0).unsqueeze(dim=0)
        changed_hidden_state = torch.mean(changed_hidden_states, dim=1).unsqueeze(dim=1)
        encoder_outputs.last_hidden_state = changed_hidden_state

        self.avg_encoder_outputs = encoder_outputs
        return encoder_outputs

    def get_avg_sentence(self, encoder_outputs=None, device="cuda"):
        self.device = device
        self.model = self.model.to(device)
        preds = self.model.generate(
            None,
            do_sample=True,
            top_p=0.9,
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

        mean_phrase = self.postprocess_phrases([list(paraphrases)[0]])[0]
        self.mean_phrase = self.wordnet_merge_similar_words(mean_phrase)
        return self.mean_phrase

    def generate_from_latent(self, encoder_outputs=None, num_candidates=512, device="cuda"):
        self.device = device

        self.model = self.model.to(device)
        if encoder_outputs:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.to(device)

        preds = self.model.generate(
            None,
            do_sample=True,
            max_length=12,
            top_k=256,
            top_p=0.9,
            early_stopping=False,
            num_return_sequences=num_candidates,
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

    @staticmethod
    def remove_empty_phrases(phrases):
        return [phrase for phrase in phrases if phrase]

    @staticmethod
    def remove_words_without_content(caption):
        stop_words = set(STOPWORDS).union(set(stopwords.words('english')))
        caption = " ".join([word for word in caption.split() if word not in stop_words])
        return " ".join([word for word in caption.split() if wn.synsets(word) and len(word) > 2])

    def remove_mean_sentence_words(self, caption):
        return " ".join([word for word in caption.split() if word not in self.mean_phrase.split()])

    def postprocess_phrases(self, phrases: list[str]):
        phrases = list(map(lambda x: self.remove_words_without_content(x), phrases))
        if self.mean_phrase:
            phrases = list(map(lambda x: self.remove_mean_sentence_words(x), phrases))
        phrases = self.remove_empty_phrases(phrases)
        phrases = list(set(phrases))
        return phrases

    def encode_text(self, input_phrases, texts_encoder=None, tokenizer=None):

        if not texts_encoder:
            texts_encoder = self.texts_encoder
        if not tokenizer:
            tokenizer = self.tokenizer

        input_phrases = [re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase) for input_phrase in input_phrases]
        input_phrases = ["paraphrase: " + input_phrase for input_phrase in input_phrases]
        input_ids = \
            tokenizer.batch_encode_plus(input_phrases, padding=True, truncation=True,
                                        return_tensors='pt').data[
                "input_ids"]
        return texts_encoder(input_ids.to(texts_encoder.base_model.device))

    def project_phrases_for_matching(self, texts: list[str], batch_size, texts_encoder=None, tokenizer=None,
                                     device="cuda"):
        """This is a helper function for weaker GPUs
        To support many kinds of embedders"""

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        if not texts_encoder:
            texts_encoder = self.texts_encoder
        if not tokenizer:
            tokenizer = self.tokenizer

        acc = []
        texts_encoder = texts_encoder.to(device)
        for chunk in chunks(texts, batch_size):
            with torch.no_grad():
                encoder_outputs = self.encode_text(chunk, texts_encoder, tokenizer)
                projected_data = torch.mean(encoder_outputs.last_hidden_state, dim=1)
                acc.append(projected_data.to("cpu"))
                torch.cuda.empty_cache()
        texts_projections = torch.cat(acc).to(device)
        texts_projections = texts_projections / torch.norm(texts_projections, dim=1, keepdim=True)
        return texts_projections

    def generate_principal_phrases(self, shorten_texts: List[str], num_of_phrases: int, mean_phrase,
                                   texts_projections=None,
                                   texts_encoder=None, mean_encoder_outputs=None,
                                   variance_coefficient=1.0, orthogonality_coefficient=-10.0,
                                   device="cuda"):
        """We allow injecting encoder, text projections etc. for efficiency"""
        self.device = device
        if texts_encoder is None:
            texts_encoder = self.texts_encoder
        if texts_projections is None:
            if self.texts_projections is None:
                self.texts_projections = self.project_phrases_for_matching(shorten_texts, 20, texts_encoder)
            texts_projections = self.texts_projections
        if mean_encoder_outputs is None:
            mean_encoder_outputs = self.get_avg_embedding(shorten_texts)

        # Mean Sentence
        mean_phrase_encoding = self.project_phrases_for_matching([mean_phrase], 1, texts_encoder)
        # Text projections
        texts_projections = texts_projections - mean_phrase_encoding

        candidate_phrases = self.generate_from_latent(encoder_outputs=mean_encoder_outputs)
        candidate_phrases = self.postprocess_phrases(candidate_phrases)
        candidate_phrases_embeddings = self.project_phrases_for_matching(candidate_phrases, 20, texts_encoder)
        for i in range(num_of_phrases):
            # Var
            variance = self.calc_variance_per_suggestion(candidate_phrases_embeddings, texts_projections)

            # Ortho
            if not self.prev_principal_phrases:
                ortho = torch.zeros(len(candidate_phrases)).to(self.device)
            else:
                previous_phrases_embeddings = \
                    self.project_phrases_for_matching(self.prev_principal_phrases, 20, texts_encoder)
                ortho = self.get_orthogonal_delta_penalty(candidate_phrases_embeddings, previous_phrases_embeddings)
            # Score
            score = variance_coefficient * variance + orthogonality_coefficient * ortho
            max_index = torch.argmax(score).item()
            self.prev_principal_phrases.append(candidate_phrases[max_index])
        return self.postprocess_phrases(self.prev_principal_phrases)

    def calc_variance_per_suggestion(self, candidate_sentences_embeddings, texts_projections):
        normed_similarity_scores = texts_projections @ candidate_sentences_embeddings.T
        mu = torch.sum(normed_similarity_scores, dim=0) / normed_similarity_scores.size(0)
        return torch.sum(torch.square(normed_similarity_scores - mu), dim=0)


    def get_orthogonal_delta_penalty(self, candidate_sentences_embeddings, previous_phrases_embeddings):
        # We want the orthogonality penalty to be depended on the current sentence only
        prev_penalty = self.calc_orthogonality_penalty(previous_phrases_embeddings.to(self.device))
        penalty_per_candidate = []
        for i in range(candidate_sentences_embeddings.shape[0]):
            vectors = torch.vstack([candidate_sentences_embeddings[i].unsqueeze(0), previous_phrases_embeddings])
            ortho_penalty = self.calc_orthogonality_penalty(vectors)
            penalty_per_candidate.append(ortho_penalty - prev_penalty)
        penalty_per_candidate = torch.tensor(penalty_per_candidate).to(self.device)
        return penalty_per_candidate


    def calc_orthogonality_penalty(self, vectors):
        count_vectors = vectors.size(0)
        eye = torch.eye(count_vectors).to(self.device)
        return torch.dist(vectors @ vectors.T, eye)


    def wordnet_merge_similar_words(self, mean_phrase):
        og_mean_phrase = mean_phrase
        used_words = []
        common_ancestors = {}
        for word_i in og_mean_phrase.split():
            if word_i in used_words: continue  # we want to enable more fine words after merger
            if len(word_i) <= 2: continue  # If not a word
            # sort indices
            get_score_f = self.get_fine_similarity_func_from_word(word_i)
            sorted_words = sorted(og_mean_phrase.split(),
                                          key=lambda x: get_score_f(x),
                                          reverse=True)
            for word_k in sorted_words:
                if word_k in used_words: continue
                if word_i == word_k: continue
                if len(word_k) <= 2: continue
                if self.wordent_shortest_path_distance(word_i, word_k) <= 2:
                    common_ancestor = self.wordnet_get_two_words_hyper(word_i, word_k)
                    common_ancestors[word_i] = common_ancestor
                    used_words.append(word_k)
                    used_words.append(word_i)
                    used_words.append(common_ancestor)
        new_sentence = []
        for word in og_mean_phrase.split():
            if word in common_ancestors.keys():
                new_sentence.append(common_ancestors[word])
                continue
            if word in used_words:
                continue
            new_sentence.append(word)
        return " ".join(new_sentence)

    @staticmethod
    def get_fine_similarity_func_from_word(og_word):
        def f(word_to_compare):
            sl1 = wn.synsets(og_word)
            sl2 = wn.synsets(word_to_compare)
            max_score = 0
            for word1 in sl1:
                for word2 in sl2:
                    if word1.name().split(".")[1] != word2.name().split(".")[1]: continue
                    similarity = word1.lch_similarity(word2)
                    if similarity is not None:
                        max_score = max(max_score, similarity)
            return max_score

        return f

    @staticmethod
    def wordnet_get_two_words_hyper(word1, word2):
        sl1 = wn.synsets(word1)
        sl2 = wn.synsets(word2)
        dist = 1000
        common_ancestor = ""
        for word1 in sl1:
            for word2 in sl2:
                path_dist = word1.shortest_path_distance(word2)
                if path_dist is None: continue
                if path_dist < dist:
                    common_ancestor = word1.lowest_common_hypernyms(word2)[0]
                    dist = word1.shortest_path_distance(word2)
        return common_ancestor.name().split(".")[0]

    @staticmethod
    def wordent_shortest_path_distance(word1, word2):
        sl1 = wn.synsets(word1)
        sl2 = wn.synsets(word2)
        min_path = 1000
        for word1 in sl1:
            for word2 in sl2:
                dist = word1.shortest_path_distance(word2)
                if dist is not None:
                    min_path = min(min_path, dist)
        return min_path

    def shorten_augment(self, input_phrase, use_gpu=False, diversity_ranker="levenshtein",
                        max_length=16, adequacy_threshold=0.90, fluency_threshold=0.90):
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