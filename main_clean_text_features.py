
import numpy as np
import pandas as pd
from functools import partial

## Hyperparameters

perc = 99 # activation percentile

verbose_iterations = 500 # avisame cada x iteraciones

## Output_paths

output_path = '/home/cmramirez/Desktop/Python/PAPER_dic_2023/3_EXTRACT_FEATURES/all_features'

## Functions

def get_activated_columns(row, df):
    activated_columns = list(df.columns[row == 1])
    return activated_columns

def process_mid(text):

    # Es para los mids
    text = text.replace(',','')
    text = text.replace('[','')
    text = text.replace(']','')
    text = text.replace("'","")

    return text


#################################################################
## Class names
#################################################################
dict_path = '../audioset_tagging_cnn-master/metadata/class_labels_indices.csv'

dt_clases = pd.read_csv(dict_path)
dict_nombres = dict(zip(dt_clases['display_name'], dt_clases['mid']))

#################################################################
## PANNs
#################################################################
dt_panns_train = pd.read_csv('../3_EXTRACT_FEATURES/fold_1_features/train/raw_panns/dt_raw_panns.csv')
dt_panns_test = pd.read_csv('../3_EXTRACT_FEATURES/fold_1_features/test/raw_panns/dt_raw_panns.csv')

dt_panns = pd.concat([dt_panns_train, dt_panns_test], axis=0)

X_dt_panns = dt_panns.select_dtypes(include='float64')
panns_probs = X_dt_panns.to_numpy().flatten()

#################################################################
## YAMNet
#################################################################
dt_yamnet_train = pd.read_csv('../3_EXTRACT_FEATURES/fold_1_features/train/raw_yamnet/dt_raw_yamnet.csv')
dt_yamnet_test = pd.read_csv('../3_EXTRACT_FEATURES/fold_1_features/test/raw_yamnet/dt_raw_yamnet.csv')

dt_yamnet = pd.concat([dt_yamnet_train, dt_yamnet_test], axis=0)

X_dt_yamnet = dt_yamnet.select_dtypes(include='float64')
yamnet_probs = X_dt_yamnet.to_numpy().flatten()

#################################################################
## For loop
#################################################################

dt_final_panns = pd.DataFrame()
dt_final_yamnet = pd.DataFrame()

print(f"Hay {len(dt_panns['audio_name_chunk'].unique())} elementos.")

for i in range(len(dt_panns['audio_name_chunk'].unique())):

    if i % verbose_iterations == 0:
        
        print(f"Iteración {i}")

    ## Coger tramos del dataframe
    dt_aux_panns = dt_panns[dt_panns['audio_name_chunk'] == dt_panns['audio_name_chunk'].unique()[i]]
    dt_aux_yamnet = dt_yamnet[dt_yamnet['audio_name_chunk'] == dt_yamnet['audio_name_chunk'].unique()[i]]

    ## Escoger los numéricos (quitar el nombre el audio)
    X_dt_panns = dt_aux_panns.select_dtypes(include='float64')
    X_dt_yamnet = dt_aux_yamnet.select_dtypes(include='float64')

    ## Convertir a binario
    binario_panns = (X_dt_panns >= np.percentile(panns_probs, perc)).astype(int)
    binario_mid_panns = binario_panns.rename(columns=dict_nombres).copy()

    binario_yamnet = (X_dt_yamnet >= np.percentile(yamnet_probs, perc)).astype(int)
    binario_mid_yamnet = binario_yamnet.rename(columns=dict_nombres).copy()

    ## Coger las columnas activadas
    binario_panns['output'] = binario_panns.apply(partial(get_activated_columns, df=binario_panns), axis=1)
    binario_mid_panns['output'] = binario_mid_panns.apply(partial(get_activated_columns, df=binario_mid_panns), axis=1)

    binario_yamnet['output'] = binario_yamnet.apply(partial(get_activated_columns, df=binario_yamnet), axis=1)
    binario_mid_yamnet['output'] = binario_mid_yamnet.apply(partial(get_activated_columns, df=binario_mid_yamnet), axis=1)

    ## Concatenar elementos por filas
    concatenated_list_panns = []
    concatenated_list_mid_panns = []
    concatenated_list_yamnet = []
    concatenated_list_mid_yamnet = []

    for row1, row2, row3, row4 in zip(binario_panns['output'], binario_mid_panns['output'], binario_yamnet['output'], binario_mid_yamnet['output']):
        concatenated_list_panns.append(row1)
        concatenated_list_mid_panns.append(row2)
        concatenated_list_yamnet.append(row3)
        concatenated_list_mid_yamnet.append(row4)

    ## Colocar en dataframes
    lista_eventos_panns = str(concatenated_list_panns)
    lista_eventos_mid_panns = str(concatenated_list_mid_panns)

    lista_eventos_yamnet = str(concatenated_list_yamnet)
    lista_eventos_mid_yamnet = str(concatenated_list_mid_yamnet)

    dt_final_panns.loc[i,'audio_name_chunk'] = str(dt_panns['audio_name_chunk'].unique()[i])
    dt_final_panns.loc[i,'eventos_acusticos'] = lista_eventos_panns
    dt_final_panns.loc[i,'mid'] = lista_eventos_mid_panns

    dt_final_yamnet.loc[i,'audio_name_chunk'] = str(dt_yamnet['audio_name_chunk'].unique()[i])
    dt_final_yamnet.loc[i,'eventos_acusticos'] = lista_eventos_yamnet
    dt_final_yamnet.loc[i,'mid'] = lista_eventos_mid_yamnet

## Procesar mids
dt_final_panns['mid_clean'] = dt_final_panns['mid'].apply(process_mid)
dt_final_yamnet['mid_clean'] = dt_final_yamnet['mid'].apply(process_mid)

## Guardar outputs
dt_final_panns.to_csv(output_path+'/dt_panns.csv',index=False)
dt_final_yamnet.to_csv(output_path+'/dt_yamnet.csv',index=False)

