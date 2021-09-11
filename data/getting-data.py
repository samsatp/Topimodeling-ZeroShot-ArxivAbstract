from tqdm.notebook import tqdm
from preprocessing import *
import tensorflow as tf
from typing import List
import numpy as np
import json
import re
import gc
import os

gc.enable()
print(tf.__version__)

data_url = os.getenv('DATA_URL')

## RAW DATA
def get_metadata():
    with open(data_url, 'r') as f:
        for line in f:
            yield line

if not os.path.exists('train'): os.mkdir('train')
if not os.path.exists('val'): os.mkdir('val')
    
## CATEGORY MAPPING
category_map = json.load(open('mapping.json'))
category_map_val = list(category_map.values())

metadata = get_metadata()

def getContent(paper):
    '''
    `getContent()` is a function to check the quality of each data record and extract only necessary part.
    '''
    paper = json.loads(paper)
    try:
        paper["title"]
        paper["abstract"]
        cats = paper["categories"].split()
        for cat in cats:
            assert cat in category_map
    except:
        return False, False
    return paper["abstract"], cats

## STORE DATASET
categories = []    # Each paper's categories
paths = []         # Path to file abstract saved
target = []        # 0,1 

## STORE VALIDATION SET
val_categories = []
val_paths = []
val_target = []



if __name__ == '__main__':

    for ind, paper in tqdm(enumerate(metadata)):
        if ind%100 > 5: continue
        
        abstract, paperCategories = getContent(paper) 
        if not abstract: continue
        
        ## WHETHER OR NOT THIS PAPER WILL COSIDERED TRUE
        isTrue = np.random.uniform() > 0.4
        isDup = np.random.uniform() < 0.1
        
        ## IS THIS PAPER WILL BE TRAINED OR VALIDATED
        isVal = np.random.uniform() > 0.9

        ## ABSRTACT
        abstract = abstract.strip()
        abstract = re.sub("\n", "<br>", abstract)
        
        ## SAVE DATA TO DISK
        if not isVal: path = f"train/abstracts/{ind}.txt"
        else: path = f"val/val_abstracts/{ind}.txt"
        
        with open(path,"w") as f:
            f.writelines(abstract)
            
        
        ## CATEGORIES & TARGET
        if isTrue or isDup:  
            paper_cat = [category_map[e] for e in paperCategories]
            t = int(1)
        
        if not isTrue:       
            n_cat = np.random.uniform(low=1, high=8)
            paper_cat = np.random.choice(category_map_val, size=int(n_cat), replace=False)
            t = int(0)
            
        if not isVal:
            paths.append(path)
            target.append(t)
            categories.append(paper_cat)
        else:
            val_paths.append(path)
            val_target.append(t)
            val_categories.append(paper_cat)

    print("TRAIN")
    print(f'\ttotal data: {len(target)}')
    print(f'\ttarget distribution 1: {sum(target)} , 0: {len(target)-sum(target)}')
    print("VAL")
    print(f'\ttotal data: {len(val_target)}')
    print(f'\ttarget distribution 1: {sum(val_target)} , 0: {len(val_target)-sum(val_target)}')
    print("="*50)

    ## Tokenize TRAIN categories
    tokenized, vect = tokenizeTrainCategories(categories=categories)

    ## Tokenize VAL categories
    val_categories_squeeze = [' '.join(e) for e in val_categories] 

    val_tokenized = vect.texts_to_sequences(val_categories_squeeze)
    val_tokenized = tf.keras.preprocessing.sequence.pad_sequences(val_tokenized, padding='post')


    ## SAVE THINGS
    np.save('val_tokenized', val_tokenized)
    np.save('val_target', val_target)

    np.save('tokenized', tokenized)
    np.save('target', target)

    ## Embedding the sentences
    embedded_abstract = embeddingSentences(paths)
    val_embedded_abstract = embeddingSentences(val_paths)


    ## save TRAIN embedded abstracts
    if not os.path.exists('train/embedded_abstracts'): os.mkdir('train/embedded_abstracts')
    for i, e in tqdm(enumerate(embedded_abstract)):
        p = f'embedded_abstracts/emb_{i}'
        np.save(p, e.numpy())
        
    ## save VAL embedded abstracts
    if not os.path.exists('val/val_embedded_abstract'): os.mkdir('val/val_embedded_abstract')
    for i, e in tqdm(enumerate(val_embedded_abstract)):
        p = f'val_embedded_abstract/emb_{i}'
        np.save(p, e.numpy())
        
    