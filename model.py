import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Embedding
from typing import List

class UseZeroClf(tf.keras.Model):
    def __init__(self, gru_units: int, dense_units: List[int], category_vocab_size: int, emb_dim: int):
        super().__init__()
        self.gru_units = gru_units
        self.gru = GRU(units=gru_units, dropout=0.1)
        self.denses = [
            Dense(units=unit, activation='relu') for unit in dense_units
        ]
        self.n_denses = len(dense_units)
        self.final_dense = Dense(units=1)
        self.embedding = Embedding(category_vocab_size+1, emb_dim, mask_zero=True)

    
    def call(self, inputs, training):
        embedded_sentences, cat_ids = inputs        

        gru_outputs = self.gru(embedded_sentences)    # (batch_size, gru_units)

        all_emb = self.embedding(cat_ids)
        avg_emb = tf.reduce_mean(all_emb, axis=1)    # (batch_size, emb_dim)
                 
        output = tf.concat([gru_outputs, avg_emb], axis=-1)  # (batch_size, emb_dim+gru_units)
        
        for i in range(self.n_denses):
            output = self.denses[i](output)
            
        output = self.final_dense(output)
        
        return output