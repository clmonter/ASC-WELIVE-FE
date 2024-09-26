from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import difflib
import pandas as pd
import numpy as np

##############################################
## TF-IDF
##############################################

def TFIDF_formating(corpus, dt_equivalencias, datos):

    countVectorizer = CountVectorizer()
    tfIdfTransformer = TfidfTransformer(use_idf=True)

    countVectorizer.fit(corpus)

    wordCount = countVectorizer.transform(datos[f'mid_clean'])
    new_TfIdf = tfIdfTransformer.fit_transform(wordCount)

    df_tfidf_labels = []
    for feature in countVectorizer.get_feature_names_out():
        match = difflib.get_close_matches(str(feature), list(dt_equivalencias['mid']), n=1, cutoff=0.7)
        string_match = dt_equivalencias.loc[dt_equivalencias['mid'] == ''.join(match), 'display_name']
        df_tfidf_labels.append(''.join(string_match))

    df_tfidf = pd.DataFrame(new_TfIdf.todense(), columns=df_tfidf_labels)
    sparse_tfidf = df_tfidf # This is the 'sparse' tdidf matrix
    sparse_tfidf = sparse_tfidf.reindex(sorted(sparse_tfidf.columns), axis=1)

    pd.set_option('display.max_columns', None)
    sparse_tfidf.insert(0, 'audio_name_chunk', datos['audio_name_chunk'])
    
    return sparse_tfidf

##############################################
## Node2Vec from TF-IDF
##############################################

def Node2Vec_from_tfidf(df_tfidf, model, ontologia):

    df_tfidf = df_tfidf.iloc[:,1:]

    # Obtener los nombres de las columnas del archivo CSV
    column_names = df_tfidf.columns.tolist()

    # Filtrar los nodos que aparecen en el archivo CSV
    filtered_nodes = [node_id for node_id in model.wv.index_to_key if ontologia.nodes[node_id]['name'] in column_names]

    # Ordenar los nodos filtrados alfabéticamente por su nombre
    filtered_nodes_sorted = sorted(filtered_nodes, key=lambda x: ontologia.nodes[x]['name'])

    # Obtener las representaciones vectoriales de los nodos seleccionados
    node_vectors = np.array([model.wv[node_id] for node_id in filtered_nodes_sorted])

    # Ver los nombres de los nodos filtrados ordenados
    node_names_sorted = [ontologia.nodes[node_id]['name'] for node_id in filtered_nodes_sorted]

    node_vectors_reshaped = node_vectors.reshape(1, node_vectors.shape[0], node_vectors.shape[1])

    df_tfidf_reshaped = np.array(df_tfidf).reshape(df_tfidf.shape[0], df_tfidf.shape[1], 1)

    # Obtener las dimensiones de los dataframes
    dim1, dim2, dim3 = df_tfidf_reshaped.shape

    # Crear un dataframe vacío con las dimensiones requeridas
    result = np.zeros((dim1, dim2, dim3 + 5))

    # Iterar sobre las filas de los dataframes originales
    for i in range(dim1):
        for j in range(dim2):
            # Obtener el valor de df_tfidf_reshaped en la posición actual
            tfidf_value = df_tfidf_reshaped[i, j, 0]

            # Si el valor es 0, completar los 5 canales restantes con 0
            if tfidf_value == 0:
                result[i, j, :] = 0
            else:
                # Si el valor no es 0, completar los 5 canales restantes con los valores de node_vectors_reshaped
                result[i, j, :dim3] = df_tfidf_reshaped[i, j, :]
                result[i, j, dim3:] = node_vectors_reshaped[0, j, :]

    return result