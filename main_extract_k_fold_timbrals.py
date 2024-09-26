

'''
To ignore warnings, run:
python -W ignore main_extract_k_fold_timbrals.py
'''


import argparse
import pandas as pd
import os
import sys
sys.path.append('../timbral_models') 
import timbral_models as tm

###################################################
## Argumentos parser
###################################################

parser = argparse.ArgumentParser()

## Number of folds in Stratified K-fold
parser.add_argument('--n_splits', type=int, default=5,
                    help='Number of splits in the stratified cross validation')

## Checkpoint_path
parser.add_argument('--save_checkpoint_path', type=str, default = '../3_EXTRACT_FEATURES/checkpoint')

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

## GPU
parser.add_argument('--cuda', type=int, default=1,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0,
                    help='set gpu id to use (default: 0)')

## Sample rate
parser.add_argument('--fs',type=float,  default=16000,
                    help = 'Sampling rate')

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

feature_names_timbrals =  ['hardness',
                            'depth',
                            'brightness',
                            'roughness',
                            'warmth',
                            'sharpness',
                            'boominess',
                            'reverb']

###################################################
## For loop
###################################################

dt_errores = pd.DataFrame()

num_errores = 0
num_elementos_totales = 0 # Para llevar el conteo total

for i in range(1, args.n_splits + 1): # Recorrer n_folds

    print(f"-------------------------- \n Starting with folder {i} \n--------------------------")

    for sub in subfolders: # Train o test

        print(f"------------- \n {sub} set \n-------------")

        ## Cargar labels
        path_labels = f"{labels_path}/fold_{i}/{sub}_{args.audio_type}_fold_{i}.csv"
        dt_labels = pd.read_csv(path_labels)

        dt_final_timbrals = pd.DataFrame(columns = ['audio_name_chunk'] + feature_names_timbrals)
        num_elementos = 0

        print(f"There are {len(dt_labels['audio_name_chunk'])} elements")

        for j, audio_name in enumerate(dt_labels['audio_name_chunk']):

            user = dt_labels.loc[j,'user']

            file_name = audio_path+'/'+user+'/'+audio_name+'.wav'

            try:

                ## Extract timbral
                timbrals = tm.timbral_extractor(file_name, fs=args.fs)

                ###################################################
                ## Prepare datasets
                ###################################################

                dt_final_timbrals.loc[num_elementos,'audio_name_chunk'] = audio_name
                dt_final_timbrals.loc[num_elementos,feature_names_timbrals] = timbrals
                
                num_elementos = num_elementos + 1
                num_elementos_totales = num_elementos_totales + 1

                ## Establecemos un checkpoint para ir guardando los datos
                if args.establish_checkpoint == True:
                    if num_elementos_totales % args.checkpoint_interations == 0:
                        try:
                            # Guardar el DataFrame en un archivo CSV con un nombre que incluya el número de iteración
                            dt_final_timbrals.to_csv(f"{args.save_checkpoint_path}/dt_timbrals_checkpoint_{sub}_{i}.csv", index=False)
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
                            dt_errores.to_csv(f"{current_path}/errors/dt_errores_timbrals.csv", index=False)
                            print(f"Error guardado en la iteración {num_elementos}")

                        except Exception as e:
                            print(f"Error al guardar el checkpoint en la iteración {num_elementos}: {str(e)}")
                            continue

                continue

        dt_final_timbrals.to_csv(f"{current_path}/fold_{i}_features/{sub}/dt_timbrals.csv",index=False)

if os.path.exists('../errors'):

    files_in_directory = os.listdir('../errors')
    
    if not files_in_directory:
        print('SUCCESFULLY FINISHED :)!!!')
    else:
        print('FINISHED!! Errors may occur. Please, verify the error folder.')
