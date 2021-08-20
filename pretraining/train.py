import numpy as np
import _pickle as pickle
import pandas as pd

from cpt import *
from DataGenerator import *

print("------LOADING DATA------")
path = "../data/seqs/"

pid_seq = pickle.load(open(path+"pid_seq","rb"))
age_seq = pickle.load(open(path+"age_seq","rb"))
sex_seq = pickle.load(open(path+"sex_seq","rb"))

code_seq = pickle.load(open(path+"code_seq","rb"))
date_seq = pickle.load(open(path+"date_seq","rb"))
util_seq = pickle.load(open(path+"type_seq","rb"))

print("------LOADING DIC------")
path = "../data/cat/"

diag2cat = pickle.load(open(path+"diag2cat","rb"))
proc2cat = pickle.load(open(path+"proc2cat","rb"))
drug2cat = pickle.load(open(path+"drug2cat","rb"))

print("# of patients: ", len(pid_seq))

MAX_VISIT=30
MAX_CODE=10
MAX_DEMO=2
PATIENT_DIM=100

BATCH_SIZE = 100
TRAIN_RATIO = 0.8
DATA_SIZE = len(age_seq)
EPOCHS = 1000

code_feature, code2int = process_code(code_seq)
util_feature, util2int = process_util(util_seq)
demo_feature, demo2int = process_demo(age_seq, sex_seq)

date_feature = date_seq

cat_seq = get_cat(code_seq, diag2cat, proc2cat, drug2cat)
cat_feature, cat2int = process_code(cat_seq)  

pickle.dump([code2int, util2int, demo2int, cat2int], open("vocabs/vocabs","wb"))

params = {
    'seqs':[demo_feature, code_feature, util_feature, date_feature, cat_feature],
    'vocab_sizes': [len(code2int), len(cat2int)],
    'batch_size':100,
    'max_visit':MAX_VISIT, 
    'max_code':MAX_CODE,
}

from sklearn.model_selection import train_test_split
train_IDs, valid_IDs = train_test_split(range(DATA_SIZE), train_size=TRAIN_RATIO, random_state=42)
train_generator = DataGenerator(list_IDs=train_IDs, shuffle=True, **params)
valid_generator = DataGenerator(list_IDs=valid_IDs, shuffle=False, **params)

m = model(
    patient_dim=PATIENT_DIM,
    max_visit=MAX_VISIT,
    max_code=MAX_CODE,
    max_demo=MAX_DEMO,
    code_vocab=len(code2int),
    demo_vocab=len(demo2int),
    util_vocab=4,
    date_vocab=366,
    cat_vocab=len(cat2int),
)

# m.compile(optimizer='RMSprop', loss=model_losses, metrics=model_metrics, loss_weights=model_weights)
m.compile(optimizer='RMSprop', loss=model_losses, loss_weights=model_weights)
print(m.summary())

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min', restore_best_weights=True)
m.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    verbose=2,
    callbacks=[earlyStopping],
)

print("------SAVING MODEL-------")
m.save('./saveModel/')







