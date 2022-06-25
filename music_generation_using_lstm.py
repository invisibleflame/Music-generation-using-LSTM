# -*- coding: utf-8 -*-
"""Music generation using LSTM
"""

import mido
import numpy as np
from mido import Message, MidiFile, MidiTrack
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence

mid = MidiFile(
    "/content/input.mid"
)
notes = []

notes = []
for msg in mid:
    if not msg.is_meta and msg.channel == 0 and msg.type == "note_on":
        data = msg.bytes()
        notes.append(data[1])

notes = np.array(notes)
print(notes.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array(notes).reshape(-1, 1))
notes = list(scaler.transform(np.array(notes).reshape(-1, 1)))

# LSTM layers requires that data must have a certain shape
# create list of lists fist
notes = [list(note) for note in notes]

# subsample data for training and prediction
X = []
y = []
# number of notes in a batch
n_prev = 30
for i in range(len(notes) - n_prev):
    X.append(notes[i : i + n_prev])
    y.append(notes[i + n_prev])
# save a seed to do prediction later
X_test = X[-300:]
X = X[:-300]
y = y[:-300]

model = Sequential()
model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("linear"))
optimizer = Adam(learning_rate=0.0001)
model.compile(loss="mse", optimizer=optimizer)

model.fit(np.array(X), np.array(y), 128, epochs=5, verbose=1)

prediction = model.predict(np.array(X_test))
prediction = np.squeeze(prediction)
prediction = np.squeeze(scaler.inverse_transform(prediction.reshape(-1, 1)))
prediction = [int(i) for i in prediction]

mid = MidiFile()
track = MidiTrack()
t = 0
for note in prediction:
    msg_on = Message.from_dict({'type': 'note_on', 'channel': 0, 'note': note, 'velocity': 67, 'time':0})
    # you need to add some pauses "note_off"
    msg_off = Message.from_dict({'type': 'note_off', 'channel': 0, 'note': note, 'velocity': 67, 'time':64})
    track.append(msg_on)
    track.append(msg_off)
    track.append(msg_off)
mid.tracks.append(track)
mid.save("music.mid")

