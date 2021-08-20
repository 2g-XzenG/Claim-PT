import tensorflow as tf
import pandas as pd
import numpy as np

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
        demo_feature, code_feature, util_feature, date_feature, cat_feature = self.seqs
        batch_demo, batch_code, batch_util, batch_date, batch_cat = [], [], [], [], [] 
        for i, ID in enumerate(list_IDs_temp):
            batch_demo.append(demo_feature[ID])
            batch_code.append(code_feature[ID])
            batch_util.append(util_feature[ID])
            batch_date.append(date_feature[ID])
            batch_cat.append(cat_feature[ID])
        
        batch_demo_feature = np.array(batch_demo)
        batch_code_feature = self.code_padding(batch_code)
        batch_util_feature = self.date_padding(batch_util)
        batch_date_feature = self.date_padding(batch_date)
        
        batch_cat_label = self.cat_labelling(batch_cat)
        batch_code_label = self.code_labelling(batch_cat)  # predict cat instead
        
        dic = (
            {
                'demo_feature': batch_demo_feature,
                'code_feature': batch_code_feature,
                'util_feature': batch_util_feature,
                'date_feature': batch_date_feature,
            },
            {
                'code_label': batch_code_label,
                'cat_label': batch_cat_label,
                'cls_label': np.zeros(batch_cat_label.shape)
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
    
    def code_labelling(self, seq):
        seq = [x[-1] for x in seq]
        
        X = np.zeros((len(seq), self.cat_vocab))
        for i, claim in enumerate(seq):
            for c in claim:
                X[i][c] = 1
        return X

    def cat_labelling(self, seq):
        seq = [x[:-1] for x in seq]
        
        X = np.zeros((len(seq), self.max_visit, self.cat_vocab))
        for i, p in enumerate(seq):
            if len(p) > self.max_visit: 
                p = p[:self.max_visit]
            for j, claim in enumerate(p):
                for c in claim:
                    X[i][j][c] = 1
        return X

def process_code(seq, PAD=True):
    new_seq = []
    if PAD: vocab2int = {"PAD":0}
    else: vocab2int = {}
    for p in seq:
        new_p = []
        for v in p:
            new_v = []
            for c in v:
                if c not in vocab2int: vocab2int[c] = len(vocab2int)
                new_v.append(vocab2int[c])
            new_p.append(new_v)
        new_seq.append(new_p)
    return new_seq, vocab2int

def process_util(seq):
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
    return new_seq, vocab2int
    
def process_demo(age_seq, sex_seq):
    new_seq = []
    vocab2int = {}
    for age, sex in zip(age_seq, sex_seq):
        p = []
        if age not in vocab2int: vocab2int[age] = len(vocab2int)
        if sex not in vocab2int: vocab2int[sex] = len(vocab2int)
        p.append(vocab2int[age])
        p.append(vocab2int[sex])
        new_seq.append(p)
    return np.array(new_seq), vocab2int

def get_cat(seq, diag2cat, proc2cat, drug2cat):
    new_seq = []
    for p in seq:
        new_p = []
        for v in p:
            new_v = []
            for c in v:
                if c in diag2cat :
                    new_c = diag2cat[c]
                elif c in proc2cat:
                    new_c = proc2cat[c]
                elif c in drug2cat:
                    new_c = drug2cat[c]
                else:
                    new_c = c[:5]
                new_v.append(new_c)
            new_p.append(new_v)
        new_seq.append(new_p)
    return new_seq




