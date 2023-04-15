from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
import torch
from torch.utils.data import DataLoader
import re
from transformers import pipeline
import dill
import os


def summarize_with_facebook(texts_to_summarize, file_name=None):
    facebook_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    if file_name:
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                shorten = dill.load(f)
                return shorten
    texts_to_summarize = [text[:1024] for text in texts_to_summarize]
    shorten = facebook_summarizer(texts_to_summarize, max_length=64, min_length=0, do_sample=False)
    shorten = [shorten_phrase["summary_text"] for shorten_phrase in shorten]

    if file_name:
        with open(file_name, "wb") as f:
            dill.dump(shorten, f)

    return shorten
