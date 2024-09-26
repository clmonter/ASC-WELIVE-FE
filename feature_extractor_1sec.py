import librosa
import pandas as pd
import numpy as np
from auxiliary_functions_1sec import reshape1sec
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern

########################################################
##  Extract features per second
########################################################

class NumericalFeatureExtractor_1sec:
    def __init__(self, args):

        # Argumentos relativos a la extracción de características básicas
        self.step = args.step
        self.n_mfcc = args.n_mfcc
        self.NFFT = args.NFFT
        self.win = args.win

        # Feature names related to MFCCs
        feature_names_mean_mfccs = [f'mean_MFCC_{i}' for i in range(self.n_mfcc)]
        feature_names_std_mfccs = [f'std_MFCC_{i}' for i in range(self.n_mfcc)]
        self.feature_names_mfccs = feature_names_mean_mfccs + feature_names_std_mfccs

        # Feature names related to librosa
        self.features_names = ['mean_RMS', 'mean_ZCR', 'mean_Centroid', 'mean_Rolloff', 'mean_Flatness', 'mean_Pitch',
                          'std_RMS', 'std_ZCR', 'std_Centroid', 'std_Rolloff', 'std_Flatness', 'std_Pitch']

    def extract_MFCCs(self,y,fs):

        seconds = int(len(y)/fs)
        
        # MFCCs
        raw_mfcc = librosa.feature.mfcc(y=y, sr=fs, hop_length=self.step, n_mfcc=self.n_mfcc, n_fft=self.NFFT)

        mean_mfcc, std_mfcc = reshape1sec(raw_mfcc, seconds)
        #mean_mfcc = raw_mfcc.mean(axis=1)
        #std_mfcc = raw_mfcc.std(axis=1)

        dt_mean_mfcc = pd.DataFrame(mean_mfcc.T)
        dt_std_mfcc = pd.DataFrame(std_mfcc.T)

        dt_mean_std_mfcc = pd.concat([dt_mean_mfcc, dt_std_mfcc], axis=1)
        dt_mean_std_mfcc.columns = self.feature_names_mfccs

        return dt_mean_std_mfcc

    def extract_librosa_features(self,y,fs):

        seconds = int(len(y)/fs)
        
        # RMS
        raw_rms = librosa.feature.rms(y=y, frame_length=self.win, hop_length=self.step, center=True, pad_mode='reflect')
        #mean_rms = raw_rms.mean()
        #std_rms = raw_rms.std()
        mean_rms, std_rms = reshape1sec(np.ravel(raw_rms), seconds)

        # Zero crossing rate
        raw_zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.win, hop_length=self.step, center=True)
        #mean_zcr = raw_zcr.mean()
        #std_zcr = raw_zcr.std()
        mean_zcr, std_zcr = reshape1sec(np.ravel(raw_zcr),seconds)

        # Spectral zentroid
        raw_centroid = librosa.feature.spectral_centroid(y=y, sr=fs, n_fft=self.NFFT, \
                                            hop_length=self.step, win_length=self.win, \
                                            window='hann', center=True, \
                                            pad_mode='reflect')
        #mean_centroid = raw_centroid.mean()
        #std_centroid = raw_centroid.std()
        mean_centroid, std_centroid = reshape1sec(np.ravel(raw_centroid),seconds)

        # Spectral rolloff
        raw_rolloff = librosa.feature.spectral_rolloff(y=y, sr=fs, n_fft=self.NFFT, \
                                                hop_length=self.step, win_length=self.win, \
                                                window='hann', center=True, \
                                                pad_mode='reflect', freq=None, \
                                                roll_percent=0.95)
        #mean_rolloff = raw_rolloff.mean()
        #std_rolloff = raw_rolloff.std()
        mean_rolloff, std_rolloff = reshape1sec(raw_rolloff, seconds)

        # Spectral flatness
        raw_flatness = librosa.feature.spectral_flatness(y=y,n_fft=self.NFFT, hop_length=self.step, \
                                                win_length=self.win, window='hann', \
                                                center=True, pad_mode='reflect', \
                                                amin=1e-10, power=2.0)
        #mean_flatness = raw_flatness.mean()
        #std_flatness = raw_flatness.std()
        mean_flatness, std_flatness = reshape1sec(raw_flatness, seconds)

        # Pitch
        pitch_track, mag = librosa.piptrack(y=y, sr=fs, n_fft=self.NFFT, hop_length=self.step, \
            fmin=0.0, fmax=4000.0, threshold=0.1, \
            win_length=self.win, window='hann', center=True, pad_mode='reflect', \
            ref=None)
            # Only count magnitude where frequency is > 0
        raw_pitch = np.zeros([mag.shape[1]])
        for i in range(0,mag.shape[1]):
            index = mag[:,i].argmax()
            raw_pitch[i] = pitch_track[index,i]
        #mean_pitch = raw_pitch.mean()
        #std_pitch = raw_pitch.std()
        mean_pitch, std_pitch = reshape1sec(raw_pitch, seconds)

        dt_mean_std_features = [mean_rms[0,:], mean_zcr[0,:], mean_centroid[0,:],
                            mean_rolloff[0,:], mean_flatness[0,:], mean_pitch[0,:], 
                            std_rms[0,:], std_zcr[0,:], std_centroid[0,:],
                            std_rolloff[0,:], std_flatness[0,:], std_pitch[0,:]]
        
        dt_mean_std_features = pd.DataFrame(data = np.array(dt_mean_std_features).T, columns = self.features_names)

        print(dt_mean_std_features.shape)

        return dt_mean_std_features

########################################################
##  Extract features full audio
########################################################

class NumericalFeatureExtractor:
    def __init__(self, args):

        # Argumentos relativos a la extracción de características básicas
        self.step = args.step
        self.n_mfcc = args.n_mfcc
        self.NFFT = args.NFFT
        self.win = args.win

        # Feature names related to MFCCs
        feature_names_mean_mfccs = [f'mean_MFCC_{i}' for i in range(self.n_mfcc)]
        feature_names_std_mfccs = [f'std_MFCC_{i}' for i in range(self.n_mfcc)]
        self.feature_names_mfccs = feature_names_mean_mfccs + feature_names_std_mfccs

        # Feature names related to librosa
        self.features_names = ['mean_RMS', 'mean_ZCR', 'mean_Centroid', 'mean_Rolloff', 'mean_Flatness', 'mean_Pitch',
                          'std_RMS', 'std_ZCR', 'std_Centroid', 'std_Rolloff', 'std_Flatness', 'std_Pitch']

    def extract_MFCCs(self,y,fs):
        
        # MFCCs
        raw_mfcc = librosa.feature.mfcc(y=y, sr=fs, hop_length=self.step, n_mfcc=self.n_mfcc, n_fft=self.NFFT)
        mean_mfcc = raw_mfcc.mean(axis=1)
        std_mfcc = raw_mfcc.std(axis=1)

        total = np.concatenate((mean_mfcc, std_mfcc)) 
        total = total.reshape(1, -1)
        dt_mean_std_mfcc = pd.DataFrame(total)

        dt_mean_std_mfcc.columns = self.feature_names_mfccs

        return dt_mean_std_mfcc

    def extract_librosa_features(self,y,fs):
        
        # RMS
        raw_rms = librosa.feature.rms(y=y, frame_length=self.win, hop_length=self.step, center=True, pad_mode='reflect')
        mean_rms = raw_rms.mean()
        std_rms = raw_rms.std()

        # Zero crossing rate
        raw_zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.win, hop_length=self.step, center=True)
        mean_zcr = raw_zcr.mean()
        std_zcr = raw_zcr.std()

        # Spectral zentroid
        raw_centroid = librosa.feature.spectral_centroid(y=y, sr=fs, n_fft=self.NFFT, \
                                            hop_length=self.step, win_length=self.win, \
                                            window='hann', center=True, \
                                            pad_mode='reflect')
        mean_centroid = raw_centroid.mean()
        std_centroid = raw_centroid.std()

        # Spectral rolloff
        raw_rolloff = librosa.feature.spectral_rolloff(y=y, sr=fs, n_fft=self.NFFT, \
                                                hop_length=self.step, win_length=self.win, \
                                                window='hann', center=True, \
                                                pad_mode='reflect', freq=None, \
                                                roll_percent=0.95)
        mean_rolloff = raw_rolloff.mean()
        std_rolloff = raw_rolloff.std()

        # Spectral flatness
        raw_flatness = librosa.feature.spectral_flatness(y=y,n_fft=self.NFFT, hop_length=self.step, \
                                                win_length=self.win, window='hann', \
                                                center=True, pad_mode='reflect', \
                                                amin=1e-10, power=2.0)
        mean_flatness = raw_flatness.mean()
        std_flatness = raw_flatness.std()

        # Pitch
        pitch_track, mag = librosa.piptrack(y=y, sr=fs, n_fft=self.NFFT, hop_length=self.step, \
            fmin=0.0, fmax=4000.0, threshold=0.1, \
            win_length=self.win, window='hann', center=True, pad_mode='reflect', \
            ref=None)
            # Only count magnitude where frequency is > 0
        raw_pitch = np.zeros([mag.shape[1]])
        for i in range(0,mag.shape[1]):
            index = mag[:,i].argmax()
            raw_pitch[i] = pitch_track[index,i]
        mean_pitch = raw_pitch.mean()
        std_pitch = raw_pitch.std()


        dt_mean_std_features = pd.concat([pd.Series(mean_rms), pd.Series(mean_zcr), pd.Series(mean_centroid), 
                                        pd.Series(mean_rolloff), pd.Series(mean_flatness), pd.Series(mean_pitch),
                                        pd.Series(std_rms), pd.Series(std_zcr), pd.Series(std_centroid), 
                                        pd.Series(std_rolloff), pd.Series(std_flatness), pd.Series(std_pitch)], axis=1)
        
        dt_mean_std_features.columns = self.features_names

        return dt_mean_std_features
    
    def extract_all_librosa(self,y,fs):

        dt_MFCCs = self.extract_MFCCs(y,fs)
        dt_librosa = self.extract_librosa_features(y,fs)

        dt_final = pd.concat([dt_MFCCs, dt_librosa], axis=1)

        return dt_final

########################################################
##  Extract spectrogram features
########################################################

class SpectrogramFeatureExtractor:
    def __init__(self,args):

        ## Frecuencia de muestreo
        self.sr = args.sample_rate

        ## Filtro de suavizado
        self.smoothing_mask = np.ones((args.smoothing_param, args.smoothing_param)) / (args.smoothing_param * args.smoothing_param)

        ## Parámetros de HOG
        self.orientations = args.orientations
        self.pixels_per_cell = args.pixels_per_cell
        self.cells_per_block = args.cells_per_block
        self.block_norm = args.block_norm

        self.feature_names_hog = [f'{i}' for i in range(2048)]

        ## Parámetros de LBP
        #self.radius = args.radius
        #self.n_points = args.n_points
        #self.method = args.method

    def HOG(self, normalized_waveform):

        spectrogram, freq, t, im = plt.specgram(normalized_waveform, Fs=self.sr)

        # Resize del espectrograma con interpolación bicúbica
        resized_spectrogram = resize(spectrogram, (521, 521), mode='reflect', order=3)

        # Aplicar mean filtering con máscaras 15x15
        filtered_spectrogram = convolve2d(resized_spectrogram, self.smoothing_mask, mode='same')

        # Convertimos en imagen entre 0 y 255
        #scaled_spectrogram = (filtered_spectrogram - filtered_spectrogram.min()) / (filtered_spectrogram.max() - filtered_spectrogram.min()) * 255
        #scaled_spectrogram = scaled_spectrogram.astype(np.uint8)

        # Calcular HOG
        hog_features = hog(filtered_spectrogram, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                                    cells_per_block=self.cells_per_block, visualize=False, block_norm=self.block_norm)
        
        dt_hog = pd.DataFrame(hog_features.reshape(1, -1))
        dt_hog.columns = self.feature_names_hog
        
        return dt_hog
    
    def LBP(self, normalized_waveform):

        spectrogram, freq, t, im = plt.specgram(normalized_waveform, Fs=self.sr)

        # Aplicar mean filtering con máscaras 15x15
        filtered_spectrogram = convolve2d(spectrogram, self.smoothing_mask, mode='same')

        # Convertimos en imagen entre 0 y 255
        scaled_spectrogram = (filtered_spectrogram - filtered_spectrogram.min()) / (filtered_spectrogram.max() - filtered_spectrogram.min()) * 255
        scaled_spectrogram = scaled_spectrogram.astype(np.uint8)

        # Local Binary Pattern
        lbp_image = local_binary_pattern(scaled_spectrogram, self.n_points, self.radius, method=self.method)

        # Normaliza el histograma si es necesario
        hist, _ = np.histogram(lbp_image.ravel())#, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        return hist # devuelve el histograma de local binary pattern




