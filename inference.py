import tensorflow_hub as hub
import tensorflow as tf
from typing import List, Union
import numpy as np
import requests
import argparse
import json
import re

## Load the models
model_path = "./saved_model"
embedding_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

model = tf.keras.models.load_model(model_path)
embedding = hub.KerasLayer(embedding_url, trainable=False)

## Load the categories mapping&tokenizer
category_map = json.load(open("./data/mapping.json"))
tokenizer = json.load(open("./data/cat_tok.json"))

## Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--abstract", help="A raw query string", type=str)
parser.add_argument("-c", "--candidates", help="Candidates of categories, separates each candidate with '<sep>'", type=str)
args = parser.parse_args()


def splito(text: str):
    texts = tf.strings.split(text, sep="<br>")
    return texts


def sentence_tokenize(abstract):
    return re.sub("\n", "<br>", abstract)  


def preprocess_abstract(abstract:str):
    abstract = abstract.strip()
    abstract = sentence_tokenize(abstract)
    return splito(abstract)


def preprocess_cadidates(candidates: str):
    return candidates.split('<sep>')


def get_tokens(candidates: List[str]):
    all_candidates_tokens = []
    for cat in candidates:
        tokens = []
        cat = cat.split(' ')
        for word in cat:
            try:
                tokens.append(tokenizer[word])
            except:
                tokens.append(tokenizer['<unk>'])
        all_candidates_tokens.append(tokens)
    return all_candidates_tokens


def get_padded_sequence(emb_sent: List[List[float]], tokens: List[int], n_candidates: int):
    a = tf.keras.preprocessing.sequence.pad_sequences([emb_sent for _ in range(n_candidates)], maxlen=45, dtype='float32', padding='post')
    b = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=45, dtype='int32', padding='post')
    return a, b


if __name__ == "__main__":
    abstract = args.abstract
    candidates = args.candidates

    ## Sentences
    splited_sentences = preprocess_abstract(abstract)
    emb_sent = embedding(splited_sentences)

    ## Candidates
    candidates = preprocess_cadidates(candidates)
    print(f'candidates: {candidates}')
    n_candidates = len(candidates)
    tokens = get_tokens(candidates)
    print(f'tokens: {tokens}')

    ## Padding the inputs
    abstract, candidates = get_padded_sequence(emb_sent=emb_sent, tokens=tokens, n_candidates=n_candidates)

    ## Prediction
    y_pred_logits = model((abstract, candidates))
    y_pred_prob = tf.nn.softmax(y_pred_logits, axis=0)

    print(y_pred_prob.numpy())