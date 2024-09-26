#
# Claudia Montero Ramírez
#
# Funciones auxiliares para la extracción de características

import numpy as np
import soundfile as sf
import resampy
import pandas as pd
from functools import partial
import re
from PIL import Image
import matplotlib.pyplot as plt

##############################################
## Reshape 1 second
##############################################
# Es para las features de librosa

def reshape1sec(raw_signal, seconds):

    if len(raw_signal.shape)>=2:
        factor = int(np.floor(raw_signal.shape[1])/seconds)
        raw_signal = raw_signal[:,:factor*seconds]
        res_signal = np.reshape(raw_signal, [raw_signal.shape[0], factor, seconds])
        mean_signal = np.mean(res_signal, axis=1)
        std_signal = np.std(res_signal, axis=1)
    else:
        factor = int(np.floor(raw_signal.shape[0])/seconds)
        raw_signal = raw_signal[:factor*seconds]
        res_signal = np.reshape(raw_signal, [factor, seconds])
        mean_signal = np.reshape(np.mean(res_signal, axis=0),[1,seconds])
        std_signal = np.reshape(np.std(res_signal, axis=0),[1,seconds])
   
    return mean_signal, std_signal

##############################################
## Preprocesado de audios
##############################################
# Leer el audio
# Hacer un check de metadatos
# Convertir a modomodal y hacer un resampling a 16kHz
### Nota: Los audios de WELIVE ya están a 16kHz

def audio_preprocessing(file_name, args):
    # Read the audio data
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Checking audio metadata
    ob = sf.SoundFile(file_name)

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    if sr != args.sample_rate:
        waveform = resampy.resample(waveform, sr, args.sample_rate)

    return waveform, sr

##############################################
## Espectrogramas
##############################################

def spectrogram2image(spectrogram):

    # Normalizar espectrograma
    espectrograma_normalizado = ((spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))) * 255
    
    # Convertir a tipo de datos uint8 (requerido por PIL)
    espectrograma_normalizado = espectrograma_normalizado.astype(np.uint8) # mode L es para guardar solo un canal

    # Convertir a imagen
    imagen_espectrograma = Image.fromarray(espectrograma_normalizado.T)

    return imagen_espectrograma

def show_spectrogram(imagen_espectrograma):

    # Para ver el espectrograma con colores
    plt.imshow(imagen_espectrograma, aspect='auto', interpolation='nearest', origin='lower')

##############################################
## Post procesamiento de texto
##############################################

def get_activated_columns(row, df):
    activated_columns = list(df.columns[row == 1])
    return activated_columns


def postprocess_text(dt, perc, name):

    dt_new = pd.DataFrame(columns = ['audio_name',f'text_{name}'])

    # Coger columnas numéricas
    X_dt = dt.select_dtypes(include='float64')

    # Convertir a binario
    probs = X_dt.to_numpy().flatten()
    binario = (X_dt > np.percentile(probs, perc)).astype(int)

    # Convertir a binario
    dt_new[f'text_{name}'] = binario.apply(partial(get_activated_columns, df=binario), axis=1)
    dt_new['audio_name'] = dt['audio_name']

    # Eliminar corchetes []
    dt_new[f'text_{name}'] = dt_new[f'text_{name}'].apply(lambda x: ', '.join(map(str, x)).replace('[','').replace(']',''))

    return dt_new

########################################################
## Counter text
########################################################

def repeat_text(row):
    output = []
    for idx, cell in enumerate(row, start=1):
        if pd.notna(cell):
            words = cell.split(', ')
            repeated_words = [word.strip() for word in words for _ in range(idx)]
            output.extend(repeated_words)
    return ', '.join(output)

def process(text):

    text = re.sub(r',(\s*,)*', ',', text)
    text = text.replace(', etc.','')
    text = text.replace('etc.','')

    return text


def postprocess_counter_text(dt, percentiles, name):

    columnas = ['audio_name'] + [f'text_{name}_{i}' for i in range(1, len(percentiles)+1)]

    dt_new = pd.DataFrame(columns = columnas)
    final_dt = pd.DataFrame(columns = ['audio_name', f'text_counter_{name}'])

    # Coger columnas numéricas
    X_dt = dt.select_dtypes(include='float64')

    # Convertir a binario
    probs = X_dt.to_numpy().flatten()

    for i, perc in enumerate(percentiles):

        if i < (len(percentiles)-1) :
        
            binario = ((X_dt >= np.percentile(probs, perc)) & (X_dt < np.percentile(probs, percentiles[i+1]))).astype(int)

        elif i == (len(percentiles)-1):

            binario = (X_dt >= np.percentile(probs, perc)).astype(int)

        dt_new[f'text_{name}_{i+1}'] = binario.apply(partial(get_activated_columns, df=binario), axis=1)
        # Eliminar corchetes []
        dt_new[f'text_{name}_{i+1}'] = dt_new[f'text_{name}_{i+1}'].apply(lambda x: ', '.join(map(str, x)).replace('[','').replace(']',''))

        dt_new['audio_name'] = dt['audio_name']


    X_new = dt_new.iloc[:,1:]
    final_dt['audio_name'] = dt_new['audio_name']
    final_dt[f'text_counter_{name}'] = X_new.apply(repeat_text, axis=1)

    final_dt[f'text_counter_{name}'] = final_dt[f'text_counter_{name}'].apply(process)


    return final_dt