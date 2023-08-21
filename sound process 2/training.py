import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define the path to the data folder
data_path = './data4'

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
print("Line 21 done")

# Loop through the data folder to extract features from audio files
X = []
y = []
for subdir, dirs, files in os.walk(data_path):
    for file in files:
        file_path = os.path.join(subdir, file)
        label = os.path.basename(os.path.dirname(file_path))
        feature = extract_features(file_path)
        X.append(feature)
        y.append(label)

# Convert the data to numpy arrays
X = np.array(X)
y = np.array(y)
print("Line 37 done")
# Convert labels to numeric format
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Line 44 done")
# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(20, 299, 1), padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])
print("Line 60 done")
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Line 64 done")
# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
print("Line 67 done")
# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)

# Save the model to a file
model.save('model.h5')
print("Model Saved")