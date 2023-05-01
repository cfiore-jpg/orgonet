import math
import numpy as np
import tensorflow as tf

class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys    = K.get_shape()[1]  # window size of keys

        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        x = tf.matmul(Q, tf.transpose(K, perm=[0,2,1]))
        if self.use_mask == True:
            x += atten_mask
        x = tf.nn.softmax(x / tf.sqrt(tf.cast(K.get_shape()[1], tf.float32)))
        return x
    
    def get_config(self):
        return {"use_mask": self.use_mask}


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        self.K = tf.Variable(tf.random.truncated_normal([input_size, output_size], stddev=.1))
        self.V = tf.Variable(tf.random.truncated_normal([input_size, output_size], stddev=.1))
        self.Q = tf.Variable(tf.random.truncated_normal([input_size, output_size], stddev=.1))
        self.attn_mtx = AttentionMatrix(use_mask=self.use_mask)

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

        K = tf.tensordot(inputs_for_keys, self.K, axes = [[2], [0]])
        V = tf.tensordot(inputs_for_values, self.V, axes = [[2], [0]])
        Q = tf.tensordot(inputs_for_queries, self.Q, axes = [[2], [0]])
        x = self.attn_mtx((K, Q))
        x = tf.matmul(x, V)
        return x

    def get_config(self):
        return {"use_mask": self.use_mask}


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)

        ## TODO: Add 3 heads as appropriate and any other necessary components
        self.h1 = AttentionHead(emb_sz, int(emb_sz/3), use_mask)
        self.h2 = AttentionHead(emb_sz, int(emb_sz/3), use_mask)
        self.h3 = AttentionHead(emb_sz, emb_sz - 2*int(emb_sz/3), use_mask)
        self.dense = tf.keras.layers.Dense(emb_sz)

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

        x1 = self.h1(inputs_for_keys, inputs_for_values, inputs_for_queries)
        x2 = self.h2(inputs_for_keys, inputs_for_values, inputs_for_queries)
        x3 = self.h3(inputs_for_keys, inputs_for_values, inputs_for_queries)

        y = tf.concat([x1, x2, x3], axis=2)
        y = self.dense(y)

        return y
    
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, emb_sz, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)

        self.ff_layer = tf.keras.Sequential([tf.keras.layers.Dense(2048, activation='leaky_relu'), 
                                             tf.keras.layers.Dense(emb_sz)])
        self.self_atten         = MultiHeadedAttention(emb_sz, True)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):

        x1 = self.self_atten(inputs, inputs, inputs)
        x2 = self.layer_norm(x1 + inputs)
        x3 = self.ff_layer(x2)
        x4 = self.layer_norm(x3 + x2)
        x5 = tf.keras.activations.relu(x4)
        return x5
    

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, emb_sz, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)

        self.ff_layer = tf.keras.Sequential([tf.keras.layers.Dense(2048, activation='leaky_relu'), 
                                             tf.keras.layers.Dense(emb_sz)])
        self.self_atten         = MultiHeadedAttention(emb_sz, True)
        self.self_context_atten = MultiHeadedAttention(emb_sz, False)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, context_sequence):

        x1 = self.self_atten(inputs, inputs, inputs)
        x2 = self.layer_norm(x1 + inputs)
        x3 = self.self_context_atten(context_sequence, context_sequence, x2)
        x4 = self.layer_norm(x3 + x2)
        x5 = self.ff_layer(x4)
        x6 = self.layer_norm(x5 + x4)
        x7 = tf.keras.activations.relu(x6)
        return x7


def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]    
    depths = np.arange(depth)[np.newaxis, :]/depth 
    angle_rates = 1 / (10000**depths)               
    angle_rads = positions * angle_rates            
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.pos_encoding = positional_encoding(window_size, embed_size)

    def call(self, x):
        x = self.embedding(x)
        x *= tf.sqrt(tf.cast(self.embed_size, tf.float32))
        x += self.pos_encoding
        return x
    
    def get_config(self):
        return {"embed_size": self.embed_size}
