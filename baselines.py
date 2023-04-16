import random
import torch
from customized_parrot import ParrotTextualPCA
from utils import summarize_with_facebook
from datasets import FoodReviewsDataset, MoviesDataset, EmailsDataset, Customized20Newsgroups
import time
from pykeops.torch import LazyTensor
from typing import List
import dill


def get_pca_phrases(texts_embeds, parrot):
    pca_phrases = []
    dummy_encoder_output = parrot.encode_text("dummy")  # for shape
    principal_components = calculate_pca(texts_embeds, num_of_components=5)
    for i in range(principal_components.shape[0]):
        dummy_encoder_output.last_hidden_state = principal_components[i].unsqueeze(0).unsqueeze(0)
        pca_phrases.append(parrot_textual_pca.generate_from_latent(encoder_outputs=dummy_encoder_output,
                                                                   num_candidates=1)[0])
    pca_phrases = parrot.postprocess_phrases(pca_phrases)
    return pca_phrases


def get_kmeans_phrases(texts_embeds, parrot):
    kmeans_phrases = []
    kmeans_labels, kmeans_centroids = KMeans_cosine(texts_embeds, K=5)
    dummy_encoder_output = parrot.encode_text("dummy")
    for i in range(kmeans_centroids.shape[0]):
        dummy_encoder_output.last_hidden_state = kmeans_centroids[i].unsqueeze(0).unsqueeze(0)
        kmeans_phrases.append(parrot_textual_pca.generate_from_latent(encoder_outputs=dummy_encoder_output,
                                                                      num_candidates=1)[0])
    kmeans_phrases = parrot.postprocess_phrases(kmeans_phrases)
    return kmeans_phrases


def KMeans_cosine(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Cosine similarity metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids
    # Normalize the centroids for the cosine similarity:
    c = torch.nn.functional.normalize(c, dim=1, p=2)

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
        cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Normalize the centroids, in place:
        c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

    if verbose:  # Fancy display -----------------------------------------------
        end = time.time()
        print(
            f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c


def calculate_pca(matrix, num_of_components=5):
    """
    Calculate the principal components of a matrix
    :param matrix: a matrix of shape (n, d)
    :return: the principal components of the matrix
    """
    A = (matrix - matrix.mean(0)) / matrix.std(0)
    U, S, V = torch.pca_lowrank(A, q=num_of_components)
    return V.t()


def calc_var(phrases_embeddings, texts_projections):
    normed_similarity_scores = texts_projections @ phrases_embeddings.T
    mu = torch.sum(normed_similarity_scores, dim=0) / normed_similarity_scores.size(0)
    return torch.sum(torch.square(normed_similarity_scores - mu), dim=0)


def calc_ortho(phrases_embeddings):
    count_vectors = phrases_embeddings.size(0)
    eye = torch.eye(count_vectors).to("cuda")
    return torch.dist(phrases_embeddings @ phrases_embeddings.T, eye)


if __name__ == "__main__":
    # Setup
    torch.random.manual_seed(0)
    random.seed(0)

    parrot_textual_pca = ParrotTextualPCA()
    food_reviews_dataset = FoodReviewsDataset()
    num_samples = 100
    title = "amazon_food"
    original_texts = list(food_reviews_dataset[0:num_samples])

    # Summarize
    filename = f"shorten_phrases_facebook_{title}_{num_samples}.pkl"
    shorten_texts = summarize_with_facebook(original_texts, filename)

    # Embeds
    texts_embeds = parrot_textual_pca.project_phrases_for_matching(shorten_texts, 20)
    # KMEANS
    kmeans_phrases = get_kmeans_phrases(texts_embeds, parrot_textual_pca)
    # PCA
    regular_pca_phrases = get_pca_phrases(texts_embeds, parrot_textual_pca)
