# ASC-WELIVE-Feature-Extraction

This repo shows how the following audio features were extracted, for the WELIVE database:

* Librosa: 19 features. These are 13 Mel-Frequency Cepstral Coefficients, 2 time-domain features (Root Mean Square and Zero Crossing Rate) and 4 frequency-domain features (Spectral Centroid, Spectral Roll- off, Spectral Flatness and Pitch). The mean and standard deviation for each feature are aggregated every 5 seconds resulting in 38 features.
* Histogram of Oriented Gradients (HOG): 2048 features.
* Local Binary Pattern (LBP): 10 features.
* 
