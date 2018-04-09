import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import adam
from keras.optimizers import sgd
from keras.layers import Dropout
import pickle
from preprocessing import get_unknown_test_data

[train_seq_data, train_label, val_seq_data, val_label, test_seq_data, test_label] = pickle.load(open('preprocessed_train_val_test.pkl', 'rb'))
max_seq_length = 14
total_vocab = 5
train_data = sequence.pad_sequences(train_seq_data, maxlen=max_seq_length)
val_data = sequence.pad_sequences(val_seq_data, maxlen=max_seq_length)
test_data = sequence.pad_sequences(test_seq_data, maxlen=max_seq_length)
embedding_size = 32
model = Sequential()
model.add(Embedding(input_dim=total_vocab, output_dim=embedding_size, input_length=max_seq_length))
model.add(LSTM(10))#, recurrent_dropout=0.2))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
adam_optimizer = adam(lr=0.005)
model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
print(model.summary())
model.fit(train_data, train_label, epochs=100, validation_data=(val_data, val_label), batch_size=128, verbose=1)
scores = model.evaluate(test_data, test_label, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

unknown_test_seq_data = get_unknown_test_data()
predictions = model.predict(unknown_test_seq_data)
complete_prediction_set = []
idx = 0
for prediction in predictions:
    if prediction >= 0.5:
        complete_prediction_set.append([idx, 1])
    else:
        complete_prediction_set.append([idx, 0])
    idx += 1

headers = ['id','prediction']
pd.DataFrame(complete_prediction_set).to_csv('test_results.csv', header=headers, index=False)
print(complete_prediction_set)
