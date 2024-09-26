import sys
import os

sys.path.append('../audioset_tagging_cnn-master/pytorch') 
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import librosa
import torch
import pandas as pd

sys.path.append('../audioset_tagging_cnn-master/utils') 
from utilities import create_folder, get_filename

sys.path.append('../audioset_tagging_cnn-master/pytorch') 
from models import *
from pytorch_utils import move_data_to_device
import config

class AudioTagging:
    def __init__ (self, args):

        # Arugments & parameters
        self.sample_rate = args.sample_rate
        self.window_size = args.window_size
        self.hop_size = args.hop_size
        self.mel_bins = args.mel_bins
        self.fmin = args.fmin
        self.fmax = args.fmax
        self.model_type = args.model_type
        self.checkpoint_path = args.checkpoint_path
        self.device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

        self.patch_hop = args.patch_hop

        self.classes_num = config.classes_num
        self.labels = config.labels

    def load_model(self):
        Model = eval(self.model_type)
        model = Model(sample_rate=self.sample_rate, window_size=self.window_size, 
            hop_size=self.hop_size, mel_bins=self.mel_bins, fmin=self.fmin, fmax=self.fmax, 
            classes_num=self.classes_num)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model'])

        # Parallel
        if 'cuda' in str(self.device):
            model.to(self.device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        else:
            print('Using CPU.')

        return model
    
    def panns_inference(self, model, audio_path, sr):

        # Load audio
        (waveform, _) = librosa.core.load(audio_path, sr=sr, mono=True)

        dt_final = pd.DataFrame()

        for i in range(int(len(waveform)/(self.sample_rate*self.patch_hop))):

            tramo = waveform[int(i*self.sample_rate):int((i+1)*self.sample_rate)]
            tramo = tramo[None, :]    # (1, audio_length)
            tramo = move_data_to_device(tramo, self.device)

            # Forward
            with torch.no_grad():
                model.eval()
                batch_output_dict = model(tramo, None)

            clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
            """(classes_num,)"""

            dt = pd.DataFrame(clipwise_output).transpose()
            dt = dt.transform(lambda x: np.log2(1 + x))

            dt_final = pd.concat([dt_final, dt], axis=0)

        return dt_final, self.labels

