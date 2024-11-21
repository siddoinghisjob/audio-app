import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import model_from_json
from tensorflow import keras
from IPython.display import Audio
import json

# Load the pre-trained model
saved_model_path = 'model.json'
saved_weights_path = 'model.h5'

with open(saved_model_path, 'r') as json_file:
    json_savedModel = json.load(json_file)

model = keras.Sequential()
for layer_config in json_savedModel['config']['layers']:
    layer_class = layer_config['class_name']
    layer_params = layer_config['config']
    if layer_class == 'InputLayer':
        model.add(keras.layers.InputLayer(shape=layer_params['batch_input_shape'][1:], name=layer_params['name']))
    elif layer_class == 'LSTM':
        model.add(keras.layers.LSTM(units=layer_params['units'], activation=layer_params['activation'],
                                    return_sequences=layer_params['return_sequences'], name=layer_params['name']))
    elif layer_class == 'Dense':
        model.add(keras.layers.Dense(units=layer_params['units'], activation=layer_params['activation'],
                                     name=layer_params['name']))

model.load_weights(saved_weights_path)
model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['categorical_accuracy'])

emotions = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
}

emo_list = list(emotions.values())

def preprocess(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=15)
    log_S = librosa.power_to_db(S, ref=np.max)
    log_S = librosa.util.fix_length(log_S, size=339, axis=1)
    log_S = log_S.T
    log_S = np.expand_dims(log_S, axis=-1)
    return np.expand_dims(log_S, axis=0)

# Streamlit UI
st.title("Emotion Detection from Audio")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    temp_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(temp_path, format="audio/wav")
    st.write("Analyzing audio...")
    
    x = preprocess(temp_path)
    predictions = model.predict(x)
    pred_np = np.squeeze(predictions, axis=0)
    
    st.write("Emotion probabilities:")
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.bar(emo_list, pred_np, color='darkturquoise')
    ax.set_ylabel("Probability (%)")
    st.pyplot(fig)

    max_emo = np.argmax(predictions)
    st.write(f"Detected emotion: **{emotions.get(max_emo, 'Unknown')}**")