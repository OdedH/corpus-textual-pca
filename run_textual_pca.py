import random
import torch
from customized_parrot import ParrotTextualPCA
from datasets import FoodReviewsDataset, MoviesDataset, EmailsDataset, Customized20Newsgroups
from utils import summarize_with_facebook


if __name__ == "__main__":
    # Setup
    torch.random.manual_seed(0)
    random.seed(0)

    parrot_textual_pca = ParrotTextualPCA()

    # Dataset
    # food_reviews_dataset = FoodReviewsDataset()
    movies_dataset = MoviesDataset(genres=[], transform=lambda x: x["Synopsis"])
    # emails_dataset = EmailsDataset(categories=[], transform=lambda x: x["Content"])
    # news_dataset = Customized20Newsgroups(categories=[])

    num_samples = 100
    title = "movies_action"
    original_texts = list(movies_dataset[0:num_samples])

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
                                                                      orthogonality_coefficient=-3.0,
                                                                      # supports other projections
                                                                      texts_projections=None,
                                                                      # supports other encoders
                                                                      texts_encoder=None,
                                                                      # device="cpu"
                                                                      )

    print(f"principal phrases: {principal_phrases}")
