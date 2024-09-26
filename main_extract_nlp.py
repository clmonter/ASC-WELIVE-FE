import argparse
import os
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import nlp_functions
import json

#############################################################
## Input arguments
#############################################################

parser = argparse.ArgumentParser()

## Rutas
parser.add_argument('--path_json',type=str, default='/home/cmramirez/Desktop/Python/PAPER_dic_2023/3_EXTRACT_FEATURES/4_NLP_module/Node2Vec/ontology.json',
                    help='Ruta al archivo json de AudioSet.')
parser.add_argument('--path_ontology', type=str, default='/home/cmramirez/Desktop/Python/PAPER_dic_2023/3_EXTRACT_FEATURES/4_NLP_module/Node2Vec/ontologia.pickle',
                    help='Ruta a la ontología de audioset en modo árbol.')
parser.add_argument('--path_model', type=str, default='/home/cmramirez/Desktop/Python/PAPER_dic_2023/3_EXTRACT_FEATURES/4_NLP_module/Node2Vec/Node2Vec.bin',
                    help='Ruta del modelo Node2Vec')
parser.add_argument('--path_data', type=str, default='/home/cmramirez/Desktop/Python/PAPER_dic_2023/3_EXTRACT_FEATURES/31_SED/312_BINARIZACION',
                    help='Datos postprocesados, con texto en tipo "mid".')
parser.add_argument('--path_audioset_labels', type=str, default='/home/cmramirez/Desktop/Python/audioset_tagging_cnn-master/metadata/class_labels_indices.csv',
                    help='Ruta a las etiquetas de AudioSet.')
parser.add_argument('--output_path', type=str, default='/home/cmramirez/Desktop/Python/PAPER_dic_2023/3_EXTRACT_FEATURES/4_NLP_module',
                    help='Ruta para los outputs del dataframe.')

## Node2Vec training
parser.add_argument('--train_Node2Vec', type=bool, default=False,
                    help='Para volver a entrenar Node2Vec con las dimensiones que queramos')
parser.add_argument('--n_dim', type=int, default=5,
                    help='Número de dimensiones que queremos en Node2Vec')
parser.add_argument('--walk_length', type=int, default=30,
                    help='Number of nodes in each walk (default: 80)')
parser.add_argument('--num_walks', type=int, default=200,
                    help='Number of walks per node (default: 10)')
parser.add_argument('--window', type=int, default=10,
                    help='Maximum distance between the current and predicted word within a sentence')
parser.add_argument('--min_count', type=int, default=1,
                    help='Ignores all words with total frequency lower than this.')

## GPU
parser.add_argument('--cuda', type=int, default=1,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0,
                    help='set gpu id to use (default: 0)')

args = parser.parse_args()

if args.cuda == 1:
   os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

#############################################################
## Load data and ontology
#############################################################

## Ontology
# Leer el archivo JSON
with open(args.path_json, 'r') as file:
    data = json.load(file)

# Crear un grafo dirigido vacío
ontologia = nx.DiGraph()

# Agregar nodos al grafo
for nodo in data:
    node_id = nodo['id']
    node_name = nodo['name']
    ontologia.add_node(node_id, name=node_name)

# Agregar aristas al grafo
for nodo in data:
    node_id = nodo['id']
    child_ids = nodo['child_ids']
    for child_id in child_ids:
        ontologia.add_edge(node_id, child_id)

with open(args.path_ontology, 'wb') as f:
    pickle.dump(ontologia, f)

## Labels
dt_equivalencias = pd.read_csv(args.path_audioset_labels)

#############################################################
## TF-IDF
#############################################################

datos_yamnet = pd.read_csv(args.path_data + '/dt_yamnet_redondeo_x10.csv')
datos_panns = pd.read_csv(args.path_data + '/dt_panns_redondeo_x10.csv')

corpus = dt_equivalencias['mid']

yamnet_tfidf = nlp_functions.TFIDF_formating(corpus, dt_equivalencias, datos_yamnet)
panns_tfidf = nlp_functions.TFIDF_formating(corpus, dt_equivalencias, datos_panns)

yamnet_tfidf.to_csv(args.output_path+'/TF-IDF/yamnet_tfidf_redondeo_x10.csv',index=False)
panns_tfidf.to_csv(args.output_path+'/TF-IDF/panns_tfidf_redondeo_x10.csv',index=False)

#############################################################
## Node2Vec training
#############################################################

if args.train_Node2Vec == True:
   
    from node2vec import Node2Vec

    node2vec = Node2Vec(ontologia, dimensions=args.n_dim, walk_length=args.walk_length, num_walks=args.num_walks)
    model = node2vec.fit(window=args.window, min_count=args.min_count)

    model.save(args.path_model)
   
else:

    try:

        with open(args.path_model, 'rb') as file:
            model = pickle.load(file)
    
    except:

        from node2vec import Node2Vec
        node2vec = Node2Vec(ontologia, dimensions=args.n_dim, walk_length=args.walk_length, num_walks=args.num_walks)
        model = node2vec.fit(window=args.window, min_count=args.min_count)
        model.save(args.path_model)

#############################################################
## Node2Vec
#############################################################

yamnet_combination = nlp_functions.Node2Vec_from_tfidf(yamnet_tfidf, model, ontologia)
panns_combination = nlp_functions.Node2Vec_from_tfidf(panns_tfidf, model, ontologia)

yamnet_node2vec = yamnet_combination[:,:,1:]
panns_node2vec = panns_combination[:,:,1:]

np.save(args.output_path+'/yamnet_combination_redondeo_x10.npy', yamnet_combination)
np.save(args.output_path+'/panns_combination_redondeo_x10.npy', panns_combination)

np.save(args.output_path+'/Node2Vec/yamnet_node2vec_redondeo_x10.npy', yamnet_node2vec)
np.save(args.output_path+'/Node2Vec/panns_node2vec_redondeo_x10.npy', panns_node2vec)