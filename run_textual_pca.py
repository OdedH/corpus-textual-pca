from parrot import Parrot
import torch
import warnings
import random
import torch
from customized_parrot import ParrotTextualPCA
from datasets import FoodReviewsDataset
from transformers import pipeline
import dill
import os
from utils import remove_words_without_content, postprocess_phrases


def summarize_with_facebook(texts_to_summarize, file_name=None):
    facebook_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    if file_name:
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                shorten_phrases = dill.load(f)
                return shorten_phrases

    shorten_phrases = facebook_summarizer(texts_to_summarize, max_length=32, min_length=0, do_sample=False)
    shorten_phrases = [shorten_phrase["summary_text"] for shorten_phrase in shorten_phrases]

    if file_name:
        with open(filename, "wb") as f:
            dill.dump(shorten_phrases, f)

    return shorten_phrases


if __name__ == "__main__":
    # Setup
    torch.random.manual_seed(0)
    random.seed(0)

    parrot = ParrotTextualPCA()

    # Dataset
    food_reviews_dataset = FoodReviewsDataset()
    original_texts = list(food_reviews_dataset[0:100])

    # Summarize
    filename = "shorten_phrases_facebook_amazon_food_100.pkl"
    shorten_phrases = summarize_with_facebook(original_texts, filename)

    embeds = parrot.get_avg_embedding(shorten_phrases)
    generated_sentences = parrot.generate_from_latent(encoder_outputs=embeds)
    generated_sentences = postprocess_phrases(generated_sentences, "food")
    # generated_sentences = list(map(lambda x: remove_words_without_content(x), generated_sentences))

    print(generated_sentences)
