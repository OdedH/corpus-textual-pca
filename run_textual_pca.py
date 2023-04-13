import warnings
import random
import torch
from customized_parrot import ParrotTextualPCA
from datasets import FoodReviewsDataset, MoviesDataset, EmailsDataset, Customized20Newsgroups
from transformers import pipeline
import dill
import os


def summarize_with_facebook(texts_to_summarize, file_name=None):
    facebook_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    if file_name:
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                shorten = dill.load(f)
                return shorten
    texts_to_summarize = [text[:1024] for text in texts_to_summarize]
    shorten = facebook_summarizer(texts_to_summarize, max_length=64, min_length=0, do_sample=False)
    shorten = [shorten_phrase["summary_text"] for shorten_phrase in shorten]

    if file_name:
        with open(filename, "wb") as f:
            dill.dump(shorten, f)

    return shorten


if __name__ == "__main__":
    # Setup
    torch.random.manual_seed(0)
    random.seed(0)

    parrot_textual_pca = ParrotTextualPCA()

    # Dataset
    # food_reviews_dataset = FoodReviewsDataset()
    # movies_dataset = MoviesDataset(genres=["action"], transform=lambda x: x["Synopsis"])
    # emails_dataset = EmailsDataset(categories=["Science"], transform=lambda x: x["Content"])
    news_dataset = Customized20Newsgroups()

    num_samples = 100
    title = "news_sport"
    original_texts = list(news_dataset[0:num_samples])

    # Summarize
    filename = f"shorten_phrases_facebook_{title}_{num_samples}.pkl"
    shorten_texts = summarize_with_facebook(original_texts, filename)

    # Embeds
    # to use a different encoder use this function
    # texts_embeds = parrot_textual_pca.project_phrases_for_matching(shorten_phrases, 20,
    #                                                                texts_encoder=None)
    mean_embeds = parrot_textual_pca.get_avg_embedding(shorten_texts)

    # mean sentence
    mean_phrase = parrot_textual_pca.get_avg_sentence(mean_embeds)
    print(f"mean sentence: {mean_phrase}")

    # Principal Phrases
    principal_phrases = parrot_textual_pca.generate_principal_phrases(shorten_texts=shorten_texts, num_of_phrases=5,
                                                                      mean_phrase=mean_phrase,
                                                                      variance_coefficient=1,
                                                                      orthogonality_coefficient=-10.0,
                                                                      # supports other projections
                                                                      texts_projections=None,
                                                                      # supports other encoders
                                                                      texts_encoder=None)

    print(f"principal phrases: {principal_phrases}")
