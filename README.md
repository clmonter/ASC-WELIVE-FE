# ASC-WELIVE-Feature-Extraction

This repo shows how the following audio features were extracted, for the WELIVE database:

* Librosa: 19 features. These are 13 Mel-Frequency Cepstral Coefficients, 2 time-domain features (Root Mean Square and Zero Crossing Rate) and 4 frequency-domain features (Spectral Centroid, Spectral Roll- off, Spectral Flatness and Pitch). The mean and standard deviation for each feature are aggregated every 5 seconds resulting in 38 features.
* Histogram of Oriented Gradients (HOG): 2048 features.
* Local Binary Pattern (LBP): 10 features.
* Timbral: 8 timbral features are extracted based on the framework outlined in AudioCommons’ Deliverable D5.8: https://github.com/AudioCommons/timbral_models
* PANNs CNN14 Audio Tagging: https://github.com/qiuqiangkong/audioset_tagging_cnn
* YAMNet Audio Tagging: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
* PANNs CNN14 TF-IDF representation
* YAMNet TF-IDF representation
* PANNs CNN14 Node2Vec representation
* YAMNet Node2Vec representation

For 5-second audio clips, the pipeline would be the following:

![imagen](https://github.com/user-attachments/assets/30c5363f-0486-42d6-8cd8-5d2e922033fd)

However, for information retrieval-based representations (TF-IDF and Node2Vec) the pipeline extends:

![imagen](https://github.com/user-attachments/assets/cfca3e74-6234-4e07-806f-e9083d4704a8)

![imagen](https://github.com/user-attachments/assets/920aaf81-c071-44a8-9a9f-589c99d6e49a)




