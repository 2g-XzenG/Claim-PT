import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def create_code_mask(code_seq):
    code_mask = tf.cast(tf.math.not_equal(code_seq, 0), tf.float32)
    return code_mask[:,:,:,tf.newaxis]

def create_visit_mask(seq):
    visit_mask = tf.cast(tf.math.not_equal(seq, 0), tf.float32)
    return visit_mask[:,:]

def scaled_dot_product_attention(Q, K, V, Q_masks, K_masks):
    d_k = K.get_shape().as_list()[-1] # d_model/h

    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (h*N, T_q, T_k)
    outputs /= d_k ** 0.5

    padding_num = -1e+7
    K_masks = tf.expand_dims(K_masks, 1) # (h*N, 1, T_k)
    K_masks = tf.tile(K_masks, [1, tf.shape(Q)[1], 1]) # (h*N, T_q, T_k)
    paddings = tf.ones_like(outputs) * padding_num
    outputs = tf.where(tf.equal(K_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

    outputs = tf.nn.softmax(outputs)
    Q_masks = tf.expand_dims(Q_masks, -1) # (h*N, T_q, 1)
    Q_masks = tf.tile(Q_masks, [1, 1, tf.shape(K)[1]]) # (h*N, T_q, T_k)
    outputs = outputs * tf.cast(Q_masks, dtype=tf.float32)

    return tf.matmul(outputs, V) # [h*N, T_q, d_model/h]

class multihead_attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multihead_attention"):
        super(multihead_attention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.query_dense = layers.Dense(units=d_model, use_bias=False)
        self.key_dense = layers.Dense(units=d_model, use_bias=False)
        self.value_dense = layers.Dense(units=d_model, use_bias=False)
        self.add =layers.Add()
        self.norm = layers.LayerNormalization()
    
    def call(self, queries, keys, values, query_masks, key_masks):
        Q = self.query_dense(queries)
        K = self.key_dense(keys)
        V = self.value_dense(values)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0) # (h*N, T_v, d_model/h)
        query_masks = tf.tile(query_masks, [self.num_heads, 1]) # (h*N, T_q)
        key_masks = tf.tile(key_masks, [self.num_heads, 1]) # (h*N, T_k)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, query_masks, key_masks) # (h*N, T_q, d_model/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2) # (N, T_q, d_model)

        # Residual connection
        outputs = self.add([queries, outputs])
        outputs = self.norm(outputs)
        
        return outputs

class ffn(tf.keras.layers.Layer):
    def __init__(self, d_model, ffn_dim, name="ffn"):
        super(ffn, self).__init__(name=name)
        self.ffn_dim = ffn_dim
        self.dense1 = layers.Dense(units=ffn_dim, activation=tf.nn.relu, use_bias=False)
        self.dense2 = layers.Dense(units=d_model, use_bias=False)
        self.add =layers.Add()
        self.norm = layers.LayerNormalization()
    
    def call(self, inputs):
        outputs = self.dense1(inputs)
        outputs = self.dense2(outputs)
        outputs = self.add([inputs, outputs])
        outputs = self.norm(outputs)
        return outputs

def cat_recall(y_true, y_pred):
    mask_value = tf.cast(tf.not_equal(tf.reduce_sum(y_true,axis=-1), 0), tf.float32)
    true_positives = tf.cast(tf.reduce_sum(tf.multiply(tf.round(y_pred), y_true), axis=-1), tf.float32)
    possible_positives = tf.cast(tf.reduce_sum(y_true, axis=-1), tf.float32)
    values = true_positives / (possible_positives + 1e-7)
    return tf.reduce_sum(values)/tf.reduce_sum(mask_value)

def cat_loss_fun(y_true, y_pred):
    loss = tf.cast(tf.keras.losses.BinaryCrossentropy(reduction='none')(y_true, y_pred), tf.float32)
    mask = tf.cast(tf.not_equal(tf.reduce_sum(y_true,axis=-1), 0), tf.float32)
    loss = tf.multiply(loss, mask)
    # return tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def model(
    max_visit,
    max_code,
    max_demo,
    
    demo_vocab,
    code_vocab,
    date_vocab,
    util_vocab,
    cat_vocab,

    patient_dim,
    vocab_dim=100,
    model_dim=100,
    ffn_dim=100,
    num_heads=2,
    num_translayer=1,
    
    model_name="TransF"):
    
    demo = layers.Input(shape=(max_demo, ), name="demo_feature")  # max_demo = 2, age&sex
    code_seq = layers.Input(shape=(max_visit, max_code), name="code_feature") 
    util_seq = layers.Input(shape=(max_visit), name="util_feature")
    date_seq = layers.Input(shape=(max_visit), name="date_feature")

    inputs = [demo, code_seq, util_seq, date_seq]
    
    # demo embedding
    demo_emb = layers.Embedding(input_dim=demo_vocab, output_dim=vocab_dim, mask_zero=True, name='demo_embedding')(demo)
    demo_emb = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(demo_emb)     

    # code sequence
    code_mask = layers.Lambda(create_code_mask)(code_seq)
    code_emb = layers.Embedding(input_dim=code_vocab, 
                                output_dim=vocab_dim, 
                                name='code_embed')(code_seq)
    code_emb = layers.Multiply()([code_emb, code_mask])
    code_emb = tf.reduce_sum(code_emb, axis=2)     

    
    # visit mask
    visit_mask = layers.Lambda(create_visit_mask)(date_seq)
    
    # util sequence   
    util_emb = layers.Embedding(input_dim=util_vocab, output_dim=vocab_dim, mask_zero=True, name='util_embedding')(util_seq)
    util_emb = layers.Multiply()([util_emb, visit_mask[:,:,tf.newaxis]])
    
    # date sequence    
    date_emb = layers.Embedding(input_dim=date_vocab, output_dim=vocab_dim, mask_zero=True, name='date_embedding')(date_seq)
    date_emb = layers.Multiply()([date_emb, visit_mask[:,:,tf.newaxis]])

    # visit sequence
    visit_emb = layers.Add()([code_emb, date_emb, util_emb]) 

    demo_emb = tf.expand_dims(demo_emb, 1) # (N, 1, emb_size)
    demo_mask = tf.ones_like(tf.reduce_sum(demo_emb, axis=2), tf.float32) # (N, 1)
    
    for trans_layer in range(num_translayer):
        multihead = multihead_attention(model_dim, num_heads, name="multihead_attention-"+str(trans_layer))(
            queries=tf.concat([demo_emb, visit_emb], 1),                         
            keys=tf.concat([demo_emb, visit_emb], 1),
            values=tf.concat([demo_emb, visit_emb], 1),
            query_masks=tf.concat([demo_mask, visit_mask], 1),
            key_masks=tf.concat([demo_mask, visit_mask], 1)
        )
        
        visit_emb = ffn(model_dim, ffn_dim, name="ffn-"+str(trans_layer))(multihead) # (N, max_visit, emb_size)
            
        demo_emb = visit_emb[:, :1, :] # (N, 1, emb_size)
        visit_emb = visit_emb[:, 1:max_visit+1, :] # (N, max_visit, emb_size)
        
    patient_embedding = layers.Dense(patient_dim, activation=None, name="patient_embedding")(tf.squeeze(demo_emb, [1]))
    
    ##################### NVP task #####################
    # code_label = layers.Dense(units=model_dim, activation=tf.nn.sigmoid)(patient_embedding)
    code_label = layers.Dense(units=cat_vocab, activation=tf.nn.sigmoid, name='code_label')(patient_embedding)
    
    ##################### CP task #####################
    # cat_label = layers.Dense(units=model_dim, activation=tf.nn.sigmoid)(visit_emb)
    cat_label = layers.Dense(units=cat_vocab, activation=tf.nn.sigmoid, name='cat_label')(visit_emb)
   
    cls_label = layers.Dense(units=model_dim, activation=tf.nn.relu)(patient_embedding)
    cls_label = layers.Dense(units=1, activation=tf.nn.sigmoid, name="cls_label")(cls_label)
        
 
    outputs = [code_label, cat_label, cls_label]
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

model_losses = {
    "cat_label": tf.keras.losses.BinaryCrossentropy(),
    "code_label": tf.keras.losses.BinaryCrossentropy(),
    "cls_label":tf.keras.losses.BinaryCrossentropy(),
}

model_metrics = {
    "cat_label": tf.keras.metrics.Recall(top_k=30),
    "code_label": tf.keras.metrics.Recall(top_k=30),
}

model_weights = {
    "cat_label": 1,
    "code_label": 1,
    "cls_label":0,
}













    
