import sys
sys.path.append('../yamnet_export') 
from params import Params
import yamnet as yamnet_model

import pandas as pd
import soundfile as sf
import resampy
import numpy as np
from functools import partial
import re
from PIL import Image
import matplotlib.pyplot as plt

class YAMNetTagging:

    def __init__(self, args):

        self.parameters = Params()
        self.yamnet_class_dt = pd.read_csv(args.path_yamnet+'/yamnet_class_map.csv')
        self.class_names = yamnet_model.class_names(args.path_yamnet+'/yamnet_class_map.csv')

        self.path_yamnet = args.path_yamnet
        
    def load_yamnet(self):

        yamnet = yamnet_model.yamnet_frames_model(self.parameters)
        yamnet.load_weights(self.path_yamnet + '/yamnet.h5')

        return yamnet

    def audio_preprocessing(self, file_name):
        # Read the audio data
        wav_data, sr = sf.read(file_name, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

        # Checking audio metadata
        ob = sf.SoundFile(file_name)

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)

        if sr != self.parameters.sample_rate:
            waveform = resampy.resample(waveform, sr, self.parameters.sample_rate)

        return waveform, sr

    def yamnet_inference(self, yamnet, waveform):
        
        ## YAMNet inference
        scores, embeddings, spectrogram = yamnet(waveform)
        scores = scores.numpy()
        spectrogram = spectrogram.numpy()

        dt = pd.DataFrame(scores)

        # Apply the log sigmoid activation
        dt = dt.transform(lambda x: np.log2(1 + x))

        return dt, spectrogram

    




