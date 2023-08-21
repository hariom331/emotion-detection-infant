import tensorflow as tf
import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder

#load model
model = tf.keras.models.load_model('model.h5')

# Define the path to the data folder
data_path = './data4'

# Load the labels
labels = []
for subdir, dirs, files in os.walk(data_path):
    for dir in dirs:
        labels.append(dir)
le = LabelEncoder()
le.fit(labels)

# Define a function to extract MFCC features from audio files
def extract_features(file_path, max_length=299):
    with open(file_path, 'rb') as f:
        X, sample_rate = librosa.load(f, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20)
    # Pad or truncate the MFCC array to the maximum length
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    return mfccs.reshape((20, max_length, 1))



file_path = './/data4//cry//4-185575-A.ogg'
features = extract_features(file_path)


features = features.reshape((1, 20, 299, 1))


prediction = model.predict(features)


predicted_index = np.argmax(prediction)
predicted_label = le.inverse_transform([predicted_index])[0]
print('Predicted Label:', predicted_label)
