import argparse
import pandas as pd
import os
from cnn14_16k_audio_tagging_1sec import *
from yamnet_audio_tagging_1sec import * 
from feature_extractor_1sec import *
import auxiliary_functions_1sec as auxiliary_functions

###################################################
## Argumentos parser
###################################################

parser = argparse.ArgumentParser()

## Number of folds in Stratified K-fold
parser.add_argument('--n_splits', type=int, default=5,
                    help='Number of splits in the stratified cross validation')

## Save spectrogram
parser.add_argument('--save_spectrogram', type= bool, default=False)

## GPU
parser.add_argument('--cuda', type=int, default=1,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0,
                    help='set gpu id to use (default: 0)')

## Path YAMNet & PANNs models
parser.add_argument('--path_yamnet', type=str, default='.../yamnet_export', 
                    help = 'Path YAMNet')
parser.add_argument('--panns_path', type=str, default = '.../audioset_tagging_cnn-master',
                    help = 'Path PANNs')

## Checkpoint_path
parser.add_argument('--save_checkpoint_path', type=str, default = '/home/cmramirez/Desktop/Python/PAPER_dic_2023/3_EXTRACT_FEATURES/checkpoint')

## Checkpoint params
parser.add_argument('--establish_checkpoint', type=bool, default=True,
                    help='Si queremos ir guardando los datos')
parser.add_argument('--checkpoint_interations', type=int, default=500,
                    help='Cada cuantos datos queremos guardarlo')
parser.add_argument('--checkpoint_errors', type=int, default=1,
                    help='Cada cuanto queremos guardar los errores')

## Tipo de audio que vamos a utilizar para extraer las características
parser.add_argument('--audio_type', type=str, default='chunk',choices = ['chunk','audio1min'],
                    help='Realmente está implementado para chunk')

## Librosa basic feature extraction parameters
parser.add_argument('--step', type=int, default = int(16000*10e-3),
                     help='')
parser.add_argument('--n_mfcc', type=int, default = 13,
                     help='')
parser.add_argument('--NFFT', type=int, default = 460,
                     help='')
parser.add_argument('--win', type=int, default = int(16000*20e-3),
                     help='')

## Smoothing mask - Spectrogram based features
parser.add_argument('--smoothing_param', type=int, default=15,
                    help='Tamaño de la máscara de suavizado')

## HOG
parser.add_argument('--orientations', type=int, 
                    default=8)
parser.add_argument('--pixels_per_cell', type=np.array,
                    default=(32,32))
parser.add_argument('--cells_per_block', type=np.array,
                    default=(1,1))
parser.add_argument('--block_norm', type=str,
                    default='L2-Hys')

## LBP
parser.add_argument('--radius', type=int, default=2,
                    help='Radio para la vecindad de píxeles')
parser.add_argument('--n_points',type=int, default=8,
                    help='Número de puntos en la vecindad')
parser.add_argument('--method', type=str,
                    default = 'uniform')

## PANNs cnn14 16k arguments
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--window_size', type=int, default=512) #400
parser.add_argument('--hop_size', type=int, default=160)
parser.add_argument('--mel_bins', type=int, default=64)
parser.add_argument('--fmin', type=int, default=50)
parser.add_argument('--fmax', type=int, default=8000) 
parser.add_argument('--model_type', type=str, default = "Cnn14_16k") #, required=True)
parser.add_argument('--checkpoint_path', type=str, default = '/home/cmramirez/Desktop/Python/audioset_tagging_cnn-master/Cnn14_16k_mAP=0.438.pth')#required=True)
parser.add_argument('--patch_hop', type=float, default=1,
                    help='Patch hop in seconds')

args = parser.parse_args()

if args.cuda == 1:
   os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

###################################################
## Variables estáticas
###################################################

# Ruta donde vamos a extraer las features
current_path = !pwd

# Ruta donde están guardadas las etiquetas de manera estructurada
labels_path = '../2_STRATIFIED_K_FOLD'

# Audio path
if args.audio_type == 'chunk':
    audio_path = '../audios_chunk'
elif args.audio_type == 'audio1min':
    audio_path = '../audios_con_etiquetas'

# Usuarias
users = ['P007','P008','P041', 'P059','T01','T02','T03','T05','V042','V104','V110','V124','V134']

# Subcarpetas
subfolders = ['test']

# Features MFCCs & librosa
feature_names_librosa = ['mean_RMS', 'mean_ZCR', 'mean_Centroid', 'mean_Rolloff', 'mean_Flatness', 'mean_Pitch',
                  'std_RMS', 'std_ZCR', 'std_Centroid', 'std_Rolloff', 'std_Flatness', 'std_Pitch']

feature_names_mean_mfccs = [f'mean_MFCC_{i}' for i in range(args.n_mfcc)]
feature_names_std_mfccs = [f'std_MFCC_{i}' for i in range(args.n_mfcc)]
feature_names_mfccs = feature_names_mean_mfccs + feature_names_std_mfccs

###################################################
## 1. Crear subcarpetas para guardar features
###################################################

for i in range(1, args.n_splits + 1):
    ruta_nueva_carpeta = f"{current_path}/fold_{i}_features"
    if not os.path.exists(ruta_nueva_carpeta):
        try:
            os.makedirs(ruta_nueva_carpeta)
            print(f"Carpeta '{ruta_nueva_carpeta}' creada exitosamente.")
        except OSError as e:
            print(f"Error al crear la carpeta: {e}")
    else:
        print(f"La carpeta '{ruta_nueva_carpeta}' ya existe.")

    # Crear las carpetas 'train' y 'test' dentro de cada carpeta 'fold_{i}_features'
    for folder_name in ['train', 'test']:
        ruta_subcarpeta = f"{ruta_nueva_carpeta}/{folder_name}"
        if not os.path.exists(ruta_subcarpeta):
            try:
                os.makedirs(ruta_subcarpeta)
                print(f"Carpeta '{ruta_subcarpeta}' creada exitosamente.")
            except OSError as e:
                print(f"Error al crear la carpeta '{folder_name}': {e}")
        else:
            print(f"La carpeta '{ruta_subcarpeta}' ya existe.")

        # Crear la carpeta 'spectrogram' dentro de 'train' y 'test'
        ruta_subcarpeta_spectrogram = f"{ruta_subcarpeta}/spectrogram"
        if not os.path.exists(ruta_subcarpeta_spectrogram):
            try:
                os.makedirs(ruta_subcarpeta_spectrogram)
                print(f"Carpeta 'spectrogram' en '{ruta_subcarpeta}' creada exitosamente.")
            except OSError as e:
                print(f"Error al crear la carpeta 'spectrogram': {e}")
        else:
            print(f"La carpeta 'spectrogram' en '{ruta_subcarpeta}' ya existe.")

        # Crear la carpeta 'raw_panns' dentro de 'train' y 'test'
        ruta_subcarpeta_raw_panns = f"{ruta_subcarpeta}/raw_panns"
        if not os.path.exists(ruta_subcarpeta_raw_panns):
            try:
                os.makedirs(ruta_subcarpeta_raw_panns)
                print(f"Carpeta 'raw_panns' en '{ruta_subcarpeta}' creada exitosamente.")
            except OSError as e:
                print(f"Error al crear la carpeta 'raw_panns': {e}")
        else:
            print(f"La carpeta 'raw_panns' en '{ruta_subcarpeta}' ya existe.")

        # Crear la carpeta 'raw_yamnet' dentro de 'train' y 'test'
        ruta_subcarpeta_raw_yamnet = f"{ruta_subcarpeta}/raw_yamnet"
        if not os.path.exists(ruta_subcarpeta_raw_yamnet):
            try:
                os.makedirs(ruta_subcarpeta_raw_yamnet)
                print(f"Carpeta 'raw_yamnet' en '{ruta_subcarpeta}' creada exitosamente.")
            except OSError as e:
                print(f"Error al crear la carpeta 'raw_yamnet': {e}")
        else:
            print(f"La carpeta 'raw_yamnet' en '{ruta_subcarpeta}' ya existe.")

###################################################
## 2. Load models
###################################################

#########################
## YAMNet
#########################

yamnet_object = YAMNetTagging(args)
yamnet = yamnet_object.load_yamnet()
yamnet_class_dt = pd.read_csv(args.path_yamnet+'/yamnet_class_map.csv')
class_names = yamnet_model.class_names(args.path_yamnet+'/yamnet_class_map.csv')

#########################
## PANNs CNN14 16K
#########################

panns_object = AudioTagging(args)
cnn14_model = panns_object.load_model()
panns_class_names = pd.read_csv(args.panns_path+'/metadata/class_labels_indices.csv')

#########################
## Create feature extraction objects
#########################

numerical_feat_extractor = NumericalFeatureExtractor(args)

spectrogram_feat_extractor = SpectrogramFeatureExtractor(args)

###################################################
## 3. For loop
###################################################

dt_errores = pd.DataFrame()
num_errores = 0
num_elementos_totales = 0 # Para el conteo del checkpoint

for i in range(1, args.n_splits + 1): # Recorrer n_folds

    print(f"-------------------------- \n Starting with folder {i} \n--------------------------")

    #######################################
    ## Recorrer train y test
    #######################################
    for sub in subfolders: # Train o test

        print(f"------------- \n {sub} set \n-------------")

        ## Cargar labels
        path_labels = f"{labels_path}/fold_{i}/{sub}_{args.audio_type}_fold_{i}.csv"
        dt_labels = pd.read_csv(path_labels)

        #######################################
        ## Re-establecer variables
        #######################################
        num_elementos = 0

        dt_final_mfccs = pd.DataFrame(columns = ['audio_name_chunk'] + feature_names_mfccs)
        dt_final_librosa = pd.DataFrame(columns = ['audio_name_chunk'] + feature_names_librosa)

        dt_panns = pd.DataFrame()
        dt_yamnet = pd.DataFrame()

        print(f"There are {len(dt_labels['audio_name_chunk'])} elements")

        #######################################
        ## Recorrer labels dataframes
        #######################################
        for j, audio_name in enumerate(dt_labels['audio_name_chunk']):

            user = dt_labels.loc[j,'user']

            file_name = audio_path+'/'+user+'/'+audio_name+'.wav'

            try:

                ## Audio preprocessing
                waveform, sr = auxiliary_functions.audio_preprocessing(file_name, args)

                ## PANNs inference
                panns_output, panns_labels = panns_object.panns_inference(cnn14_model, file_name, args.sample_rate)
                panns_output.columns = panns_class_names['display_name'].tolist()

                ## YAMNet inference
                yamnet_output, spectrogram = yamnet_object.yamnet_inference(yamnet, waveform)
                yamnet_output.columns = yamnet_class_dt['display_name'].tolist()

                ## Extract numerical features
                dt_MFCCs = numerical_feat_extractor.extract_MFCCs(waveform,sr)
                dt_librosa = numerical_feat_extractor.extract_librosa_features(waveform,sr)

                ## Extract spectrogram based features
                hog_features = spectrogram_feat_extractor.HOG(waveform)
                lbp_histogram = spectrogram_feat_extractor.LBP(waveform)

                ### Guardar espectrograma si así se ha decidido
                if args.save_spectrogram == True:

                    # Pasar el espectrograma a formato de imagen
                    imagen_espectrograma = auxiliary_functions.spectrogram2image(spectrogram)

                    nombre_imagen = f"{current_path}/fold_{i}_features/{sub}/spectrogram/{audio_name}.png"
                    imagen_espectrograma.save(nombre_imagen, format='PNG')
                

                ###################################################
                ## Prepare datasets
                ###################################################

                ## YAMNet & PANNs
                panns_output['audio_name_chunk'] = audio_name
                yamnet_output['audio_name_chunk'] = audio_name

                dt_panns = pd.concat([dt_panns, panns_output], axis=0)
                dt_yamnet = pd.concat([dt_yamnet, yamnet_output], axis=0)

                ## MFCCs & Librosa
                dt_final_mfccs.loc[num_elementos,'audio_name_chunk'] = audio_name
                dt_final_mfccs.loc[num_elementos,feature_names_mfccs] = dt_MFCCs.iloc[0]

                dt_final_librosa.loc[num_elementos,'audio_name_chunk'] = audio_name
                dt_final_librosa.loc[num_elementos,feature_names_librosa] = dt_librosa.iloc[0]

                ## HOG & LBP
                dt_final_hog = pd.concat([dt_final_hog, pd.DataFrame([hog_features])], axis=0)
                dt_final_lbp = pd.concat([dt_final_lbp, pd.DataFrame([lbp_histogram])], axis=0)

                dt_final_hog.loc[j,'audio_name_chunk'] = audio_name
                dt_final_lbp.loc[j,'audio_name_chunk'] = audio_name


                num_elementos = num_elementos + 1

                num_elementos_totales = num_elementos_totales + 1 # Para el checkpoint

                if args.establish_checkpoint == True:
                    if num_elementos_totales % args.checkpoint_interations == 0:
                        try:
                            # Guardar el DataFrame en un archivo CSV con un nombre que incluya el número de iteración
                            dt_panns.to_csv(f"{args.save_checkpoint_path}/dt_raw_panns_checkpoint_{sub}_{i}.csv",index=False)
                            dt_yamnet.to_csv(f"{args.save_checkpoint_path}/dt_raw_yamnet_checkpoint_{sub}_{i}.csv",index=False)

                            dt_final_mfccs.to_csv(f"{args.save_checkpoint_path}/dt_mfccs_checkpoint_{sub}_{i}.csv",index=False)
                            dt_final_librosa.to_csv(f"{args.save_checkpoint_path}/dt_librosa_checkpoint_{sub}_{i}.csv",index=False)

                            print(f"Checkpoint guardado en la iteración {num_elementos}. Total {num_elementos_totales} iteraciones.")

                        except Exception as e:

                            print(f"Error al guardar el checkpoint en la iteración {num_elementos}: {str(e)}")

                            continue

            except Exception as e:

                # Por si hay algún audio de mala calidad o de corta duración, del que no se puedan extraer características
                error_message = str(e) 

                # Guardamos el error y el audio en el que se ha producido.
                dt_errores.loc[num_errores, 'audio_name'] = audio_name
                dt_errores.loc[num_errores, 'error'] = error_message

                num_errores = num_errores + 1

                ## Establecemos un checkpoint para ir guardando los errores
                if args.establish_checkpoint == True:
                    if num_errores % args.checkpoint_errors == 0:
                        try:
                            # Guardar el DataFrame en un archivo CSV con un nombre que incluya el número de iteración
                            dt_errores.to_csv(f"{current_path}/errors/dt_errores.csv", index=False)
                            print(f"Error guardado en la iteración {num_elementos}")

                        except Exception as e:
                            print(f"Error al guardar el checkpoint en la iteración {num_elementos}: {str(e)}")
                            continue

                continue


        dt_panns.to_csv(f"{current_path}/fold_{i}_features/{sub}/raw_panns/dt_raw_panns.csv",index=False)
        dt_yamnet.to_csv(f"{current_path}/fold_{i}_features/{sub}/raw_yamnet/dt_raw_yamnet.csv",index=False)

        dt_final_mfccs.to_csv(f"{current_path}/fold_{i}_features/{sub}/dt_mfccs.csv",index=False)
        dt_final_librosa.to_csv(f"{current_path}/fold_{i}_features/{sub}/dt_librosa.csv",index=False)

if os.path.exists('../errors'):

    files_in_directory = os.listdir('../errors')
    
    if not files_in_directory:
        print('SUCCESFULLY FINISHED :)!!!')
    else:
        print('FINISHED!! Errors may occur. Please, verify the error folder.')
