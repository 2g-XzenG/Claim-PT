import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, seqs, vocab_sizes, list_IDs, max_visit, max_code, batch_size=100, shuffle=True):
        self.seqs = seqs
        self.code_vocab = vocab_sizes[0]
        self.cat_vocab = vocab_sizes[1]
        self.list_IDs = list_IDs
        self.max_visit = max_visit
        self.max_code = max_code
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch' 
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        demo_feature, code_feature, util_feature, date_feature, cls_feature = self.seqs
        batch_demo, batch_code, batch_util, batch_date, batch_cls = [], [], [], [], []
        for i, ID in enumerate(list_IDs_temp):
            batch_demo.append(demo_feature[ID])
            batch_code.append(code_feature[ID])
            batch_util.append(util_feature[ID])
            batch_date.append(date_feature[ID])
            batch_cls.append(cls_feature[ID])
        
        batch_demo_feature = np.array(batch_demo)
        batch_code_feature = self.code_padding(batch_code)
        batch_util_feature = self.date_padding(batch_util)
        batch_date_feature = self.date_padding(batch_date)
        batch_cls = np.array(batch_cls)
        
        dic = (
            {
                'demo_feature': batch_demo_feature,
                'code_feature': batch_code_feature,
                'util_feature': batch_util_feature,
                'date_feature': batch_date_feature,
            },
            {
                'cls_label': batch_cls
            })
        return dic
    
    def date_padding(self, seq):
        seq = [x[:-1] for x in seq]
        
        pad_seq = np.zeros((len(seq), self.max_visit))
        for i, p in enumerate(seq):
            pad_seq[i][:len(p)] = p[:self.max_visit]
        return pad_seq
    
    def code_padding(self, seq):
        seq = [x[:-1] for x in seq]
        
        X = np.zeros((len(seq), self.max_visit, self.max_code))
        for i, p in enumerate(seq):
            if len(p) > self.max_visit: 
                p = p[:self.max_visit]
            for j, claim in enumerate(p):
                claim = claim[:self.max_code]
                X[i][j][:len(claim)] = claim
        return X

def process_code(seq, vocab2int):
    unseen = []
    new_seq = []
    for p in seq:
        new_p = []
        for v in p:
            new_v = []
            for c in v:
                if c not in vocab2int: 
                    unseen.append(c)
                    continue
                    # vocab2int[c] = len(vocab2int)
                new_v.append(vocab2int[c])
            new_p.append(new_v)
        new_seq.append(new_p)
        
    print("UNSEEN VOCAB:",len(set(unseen)), len(unseen))
    return new_seq

def process_util(seq, util2int):
    new_seq = []
    vocab2int = {"PAD":0,"IP":1,"RX":2,"OP":3}
    for p in seq:
        new_p = []
        for v in p:
            if "IP" in v:
                new_v=1
            elif "RX" in v:
                new_v=2
            else:
                new_v=3
            new_p.append(new_v)
        new_seq.append(new_p)
    return new_seq
    
def process_demo(age_seq, sex_seq, vocab2int):
    new_seq = []
    for age, sex in zip(age_seq, sex_seq):
        p = []
        assert age in vocab2int
        assert sex in vocab2int
        
        p.append(vocab2int[age])
        p.append(vocab2int[sex])
        new_seq.append(p)
    return np.array(new_seq)

def get_cat(seq,code2cat):
    new_seq = []
    for p in seq:
        new_p = []
        for v in p:
            new_v = []
            for c in v:
                new_c = code2cat[c]
                new_v.append(new_c)
            new_p.append(new_v)
        new_seq.append(new_p)
    return new_seq


print("==========LOADING DATA==========")
age_seq = pickle.load(open("../../suicideRisk/data/new_age_seq","rb"))
sex_seq = pickle.load(open("../../suicideRisk/data/new_sex_seq","rb"))

util_seq = pickle.load(open("../../suicideRisk/data/new_util_seq","rb"))
code_seq = pickle.load(open("../../suicideRisk/data/new_code_seq","rb"))
date_seq = pickle.load(open("../../suicideRisk/data/new_date_seq","rb"))
label_seq = pickle.load(open("../../suicideRisk/data/new_label_seq","rb"))

print("------LOADING DIC------")
path = "/Users/xxz005/Desktop/RAW_DATA/code2cat/"

diag2cat = pickle.load(open(path+"diag2cat","rb"))
proc2cat = pickle.load(open(path+"proc2cat","rb"))
drug2cat = pickle.load(open(path+"drug2cat","rb"))

code2cat = {**diag2cat, **proc2cat, **drug2cat}

code2int, util2int, demo2int, cat2int  = pickle.load(open("../../pretraining/model/vocabs/vocabs","rb"))

code_feature = process_code(code_seq, code2int)
util_feature = process_util(util_seq, util2int)
demo_feature = process_demo(age_seq, sex_seq, demo2int)
date_feature = date_seq

cls_feature = np.array(label_seq).reshape((-1,1))

MAX_VISIT=30
MAX_CODE=10
MAX_DEMO=2
PATIENT_DIM=100

BATCH_SIZE = 500
TRAIN_RATIO = 0.7
DATA_SIZE = len(age_seq)
EPOCHS = 20

params = {
    'seqs':[demo_feature, code_feature, util_feature, date_feature, cls_feature],
    'vocab_sizes': [len(code2int), len(cat2int)],
    'batch_size':100,
    'max_visit':MAX_VISIT, 
    'max_code':MAX_CODE,
}


generator = DataGenerator(list_IDs=range(DATA_SIZE), shuffle=False, **params)


model_path = "./saveModel"

model = tf.keras.models.load_model(model_path)
model_losses = {
    "cls_label":tf.keras.losses.BinaryCrossentropy(),
}

model_metrics = {
    "cls_label": tf.keras.metrics.AUC(),
}

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss=model_losses, metrics=model_metrics)
print(model.summary())

layer_name = 'multihead_attention-0'
intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
encoded_features = intermediate_layer_model.predict(generator)

pickle.dump(encoded_features, open("attention_weights", "wb"))
