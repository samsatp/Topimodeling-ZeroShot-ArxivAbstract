import tensorflow_hub as hub
import tensorflow as tf
from typing import List
import json
import tqdm
print(tf.__version__)

sentences_embedding = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-large/5", trainable=False)

def tokenizeTrainCategories(categories: List):
    '''
    TOKENIZATION the train categories
        return: tokenized_train_categories, tokenizer
    '''

    # Find the max number of categories of any paper
    max_len_cat = max([len(e) for e in categories])
    print(f'A paper contain at most {max_len_cat} categories')
    
    categories = [list(e) for e in categories]  # Make sure `categories` is a List[List[str]] 
    categoris_VocabRepo = []                    # categoris_VocabRepo = List[str] containing all categories in 1 level
    for c in categories:
        categoris_VocabRepo.extend(c)

    vect = tf.keras.preprocessing.text.Tokenizer(oov_token="<unk>", split=' ')
    vect.fit_on_texts(categoris_VocabRepo)      # Fit on all vocabs of categories

    

    categories_squeeze = [' '.join(e) for e in categories]  # ' '.join(all categories of each paper) into 1 long string

    tokenized = vect.texts_to_sequences(categories_squeeze)
    tokenized = tf.keras.preprocessing.sequence.pad_sequences(tokenized, padding='post')

    ## SAVE the tokenizer
    word2index = json.loads(vect.get_config()['word_index'])
    vocab_size_cat = len(word2index)                          
    print(f'Categories has vocab size of {vocab_size_cat}')

    cat_tok = json.dump(word2index, open('cat_tok.json','w'))

    return tokenized, vect


def embeddingSentences(paths: List[str]):

    splito = lambda text: tf.strings.split(text, sep="<br>")

    abstract_dataset = tf.data.TextLineDataset(paths)
    abstract_dataset = abstract_dataset.map(lambda x:splito(x))
    abstract_dataset = abstract_dataset.map(lambda x:sentences_embedding(x), num_parallel_calls=tf.data.AUTOTUNE)

    embedded_abstract = []
    for res in tqdm(abstract_dataset):
        embedded_abstract.append(res)

    max_len_abstracts = max([a.shape[0] for a in embedded_abstract])

    print(f'max_len_abstracts: {max_len_abstracts}')

    return embedded_abstract
