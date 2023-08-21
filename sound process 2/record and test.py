import pyaudio
import wave
import os
import numpy as np
import tensorflow as tf
import librosa
from pushbullet import Pushbullet
import time

# Define the path to the saved model
model_path = 'model.h5'

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Define the categories
categories = ['cry', 'laugh', 'silence']

# Define the sampling rate
RATE = 22050

# Define the number of channels
CHANNELS = 1

# Define the chunk size
CHUNK = 512

# Initialize PyAudio
p = pyaudio.PyAudio()

api_key = 'o.KXBun4UI1n9BG3AMIdR3iD2ik8f9dQMq'
pb = Pushbullet(api_key)





# Loop to continuously record and predict audio
while True:
    # Open a stream for recording
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Print a message to indicate recording has started
    print("Recording...")

    # Record audio for the specified duration
    frames = []
    for i in range(0, int(RATE / CHUNK * 5)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Print a message to indicate recording has ended
    print("Finished recording...")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Save the recorded audio to a WAV file
    wf = wave.open("recorded_audio.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Load the recorded audio file and extract the MFCC features
    with open("recorded_audio.wav", 'rb') as f:
        X, sample_rate = librosa.load(f, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20)

    # Pad or truncate the MFCC array to the maximum length
    max_length = 299
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    mfccs = mfccs.reshape((1, 20, max_length, 1))

    # Make a prediction on the MFCC features using the trained model
    prediction = model.predict(mfccs)
    category_index = np.argmax(prediction)
    category = categories[category_index]

    # Print the predicted category
    print("Predicted category:", category)

    # Push Notification
    push = pb.push_note("Update", category)

    # Send a push notification with the predicted category
    # push_message = "The predicted category is: {}".format(category)
    # pb.push_note("Audio Prediction", push_message, device)

    # Delete the recording file
    os.remove('recorded_audio.wav')
    print("Recording deleted.")

    # Wait for 0 seconds before recording again
    time.sleep(0)
