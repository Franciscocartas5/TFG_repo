# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:27:56 2019

@author: Francisco A
"""

def transcript(filewav):

    import numpy as np
    import librosa




    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers.normalization import BatchNormalization
    from keras.layers import Dropout

    # fix random seed for reproducibility
    seed = 1337
    np.random.seed(seed)

    def batch(iterable, n):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]


    def normal(stat, feature_matrix):
        """Normalize feature matrix with internal statistics of the class
        Parameters
        ----------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Feature matrix to be normalized
        Returns
        -------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Normalized feature matrix
        """

        return np.nan_to_num((feature_matrix - stat['mean']) / stat['std'])

    def normal2(matrix):

        aux  = matrix.T
        print("Normalizing...")
        for idx in batch(np.arange(aux.shape[0]), 1000):
            aux[idx, :] = normal(stat , aux[idx, :])

        return aux



    # define baseline model
    def create_model():
        # create model
        model = Sequential()
        model.add(Dense(500, input_dim=252, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(250, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(250, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(88, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        keras_weight_file = 'weights_DNN_500_250_250_MP.h5'   
        model.load_weights(keras_weight_file)
        print("MODEL LOADED ", keras_weight_file)


        return model


    def evaluate(X_test, model):

        preds = model.predict(X_test, verbose=0)
        Y_pred = 1.0 * (np.squeeze(preds) > 0.5)
        print("* * Test scores:     ", Y_pred)
        

        return Y_pred




    # LOAD DATA TEST

    File = filewav
    aux,sr = librosa.core.load( File  ,sr=44100, mono=True)
    X_test = np.abs(librosa.cqt(aux, sr=sr, hop_length=512, n_bins=252, bins_per_octave=36))

    stat = dict()
    stat['mean'] = np.loadtxt('mean_stat.txt' , delimiter = ',')
    stat['std'] = np.loadtxt('std_stat.txt' , delimiter = ',')

    X_test = normal2(X_test)


    # build the model
    model = create_model()
    #Final evaluation of the model
    Y_pred = evaluate(X_test , model)

    return Y_pred
