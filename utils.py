from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
import torch
from torch.utils.data import DataLoader
import re


def remove_words_without_content(caption):
    stop_words = set(STOPWORDS).union(set(stopwords.words('english')))
    caption = " ".join([word for word in caption.split() if word not in stop_words])
    return " ".join([word for word in caption.split() if wn.synsets(word) and len(word) > 2])


def remove_empty_phrases(phrases):
    return [phrase for phrase in phrases if phrase]


def remove_mean_sentence_words(caption, avg_sentence):
    return " ".join([word for word in caption.split() if word not in avg_sentence.split()])


def postprocess_phrases(phrases: list[str], avg_sentence=None):
    phrases = list(map(lambda x: remove_words_without_content(x), phrases))
    if avg_sentence:
        phrases = list(map(lambda x: remove_mean_sentence_words(x, avg_sentence), phrases))
    phrases = remove_empty_phrases(phrases)
    return phrases


def project_sentences_for_matching(dataset, batch_size, number_of_batches, tokenizer, encoder, device):
    acc = []
    for data in DataLoader(dataset, batch_size=batch_size):
        with torch.no_grad():
            input_phrases = [re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase) for input_phrase in data]
            input_phrases = ["paraphrase: " + input_phrase for input_phrase in input_phrases]
            input_ids = \
                tokenizer.batch_encode_plus(input_phrases, padding=True, truncation=True,
                                            return_tensors='pt').data[
                    "input_ids"]
            encoder_outputs = encoder(input_ids)
            projected_data = torch.mean(encoder_outputs.last_hidden_state, dim=1).unsqueeze(dim=1)
            acc.append(projected_data.to("cpu"))
            torch.cuda.empty_cache()
        number_of_batches -= 1
        if number_of_batches == 0:
            break
    return torch.cat(acc).to(device)
