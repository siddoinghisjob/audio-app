import os
import json
import numpy as np
import librosa
import noisereduce as nr
import tensorflow as tf
import keras
from keras.models import model_from_json
from pydub import AudioSegment, effects
import streamlit as st
import soundfile as sf
import matplotlib.pyplot as plt
import time
import tempfile

# Load model and weights
saved_model_path = 'model.json'
saved_weights_path = 'model.h5'

# Load the JSON model file
with open(saved_model_path, 'r') as json_file:
    json_savedModel = json.load(json_file)

# Manually create the model matching the original architecture
model = keras.Sequential()

# Add layers based on the saved configuration
layers_config = json_savedModel['config']['layers']
for layer_config in layers_config:
    layer_class = layer_config['class_name']
    layer_params = layer_config['config']

    if layer_class == 'InputLayer':
        input_shape = layer_params.get('batch_input_shape', [None])[1:]
        model.add(keras.layers.InputLayer(shape=input_shape, name=layer_params.get('name')))

    elif layer_class == 'LSTM':
        model.add(keras.layers.LSTM(
            units=layer_params.get('units', 64),
            activation=layer_params.get('activation', 'tanh'),
            return_sequences=layer_params.get('return_sequences', False),
            name=layer_params.get('name')
        ))

    elif layer_class == 'Dense':
        model.add(keras.layers.Dense(
            units=layer_params.get('units'),
            activation=layer_params.get('activation'),
            name=layer_params.get('name')
        ))

# Load the weights
model.load_weights(saved_weights_path)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='RMSProp',
    metrics=['categorical_accuracy']
)

# Define emotions dictionary
emotions = {
    0 : 'neutral',
    1 : 'calm',
    2 : 'happy',
    3 : 'sad',
    4 : 'angry',
    5 : 'fearful',
    6 : 'disgust',
    7 : 'surprised'
}
emo_list = list(emotions.values())

# Function to check if the data is silent
def is_silent(data):
    return max(data) < 100

# Preprocess the audio file
def preprocess(file_path, frame_length=2048, hop_length=512):
    # Fetch sample rate.
    _, sr = librosa.load(path=file_path, sr=None)
    
    # Load audio file
    rawsound = AudioSegment.from_file(file_path, duration=None)
    
    # Normalize to 5 dBFS
    normalizedsound = effects.normalize(rawsound, headroom=5.0)
    
    # Transform the audio file to np.array of samples
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
    
    # Noise reduction
    final_x = nr.reduce_noise(y=normal_x, sr=sr)

    # Extract features
    f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode='reflect').T # RMS Energy
    f2 = librosa.feature.zero_crossing_rate(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True).T # Zero Crossing Rate
    f3 = librosa.feature.mfcc(y=final_x, sr=sr, S=None, n_mfcc=13, hop_length=hop_length).T # MFCC

    # Combine features
    X = np.concatenate((f1, f2, f3), axis=1)

    # Reshape to include the batch dimension
    X_3D = np.expand_dims(X, axis=0)

    return X_3D

# Streamlit interface
st.title("Audio Emotion Analysis")

# File uploader widget for audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Create a temporary file for storing the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        file_path = tmp_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Display audio player in Streamlit
    st.audio(file_path)

    # SESSION START
    st.write(" Session started ")
    total_predictions = []  # A list for all predictions in the session.
    tic = time.perf_counter()

    # Process the audio in chunks (you may need to adjust this for your specific use case)
    y, sr = librosa.load(file_path, sr=None)
    hop_length = int(7.1 * sr)  # 7.1 seconds per chunk
    for start in range(0, len(y), hop_length):
        end = start + hop_length
        data = y[start:end]

        # Save the segmented audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_audio_path = temp_file.name
            sf.write(temp_audio_path, data, sr)

        # Display audio segment in Streamlit
        st.write(f"Playing segment from {start/sr:.2f} to {end/sr:.2f} seconds:")
        st.audio(temp_audio_path)

        # Preprocess the segment and get predictions
        x = preprocess(temp_audio_path)
        predictions = model.predict(x)
        pred_list = list(predictions)
        pred_np = np.squeeze(np.array(pred_list).tolist(), axis=0)
        total_predictions.append(pred_np)

        # Display emotion distribution for the segment
        fig = plt.figure(figsize=(10, 2))
        plt.bar(emo_list, pred_np, color='darkturquoise')
        plt.ylabel("Probability (%)")
        st.pyplot(fig)

        # Determine and display the predicted emotion
        max_emo = np.argmax(predictions)
        st.write(f"Predicted emotion: {emotions.get(max_emo, 'Unknown')}")

        # Check if the segment is silent
        if is_silent(data):
            st.write("Segment is silent. Ending session.")
            break

    # SESSION END
    toc = time.perf_counter()
    st.write("** Session ended **")

    # Present emotion distribution for the entire session
    total_predictions_np = np.mean(np.array(total_predictions).tolist(), axis=0)
    fig = plt.figure(figsize=(10, 5))
    plt.bar(emo_list, total_predictions_np, color='indigo')
    plt.ylabel("Mean probability (%)")
    plt.title("Session Summary")
    st.pyplot(fig)

    st.write(f"Emotions analyzed for: {(toc - tic):0.4f} seconds")