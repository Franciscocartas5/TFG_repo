# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:32:38 2019

@author: Francisco A
"""
import numpy as np
import os
import librosa

from normalizer import FeatureNormalizer
from normalizador import batch , acumular , normalizar
import f1_scores_func
from f1_scores_func import f1_framewise , error_tot , evalTestSet
from manageModels import save_scores_in_fig , reformat_array



from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 1337
np.random.seed(seed)
norm = FeatureNormalizer()


BBDD = []
out = []
def listfiles(path , i):
        
    lstFiles = []
    lstoutput = []
    #Lista con todos los ficheros del directorio:
    lstDir = os.walk(path)   #os.walk()Lista directorios y ficheros
     
       
    for root, dirs, files in lstDir:
        for fichero in files:
            (nombreFichero, extension) = os.path.splitext(fichero)
            if(nombreFichero[0] == 'b'):
                lstFiles.append(nombreFichero+extension)
            if(nombreFichero[0] == 'y'):
                lstoutput.append(nombreFichero+extension)
                #print (nombreFichero+extension)
                             
    print ('LISTADO FINALIZADO')
    print "longitud de la lista P" + str(i) + "= ", len(lstFiles)
    return lstFiles , lstoutput
def list_poly():
        
    for i in range (2,8):
        
        polypath = '/home/tfgcartas/code/poly/'
        path = polypath + 'P' + str(i)
        BBDD , out = listfiles(path , i)
        os.chdir(path)
        for j in range(len(BBDD)):
                
                k = 0
                while not(BBDD[j][4:] == out[j][1:]):
                  
                  print(BBDD[j] , out[j])  
                  if (BBDD[j][4:] == out[k][1:]):
                      aux = out[j]
                      out[j] = out[k]
                      out[k] = aux
                      
                  k += 1
                    
                coefs = np.loadtxt(BBDD[j] , delimiter= ',')
                outpts = np.loadtxt(out[j] , delimiter = ',')
                
                l_train = int(round(coefs.shape[1]*0.6))
                l_val_test = int(round(coefs.shape[1]*0.2))
                
                x_train = coefs[:,:l_train]
                x_val = coefs[: , l_train:(coefs.shape[1] - l_val_test)]
                x_test = coefs[: , (coefs.shape[1] - l_val_test):coefs.shape[1]]
        
                y_train = outpts[:,:l_train]
                y_val = outpts[: , l_train:(coefs.shape[1] - l_val_test)]
                y_test = outpts[: , (coefs.shape[1] - l_val_test):coefs.shape[1]]
            
                if(j<1 and i==2):
                    X_train = x_train
                    X_val = x_val
                    X_test = x_test
                    
                    Y_train = y_train
                    Y_val = y_val
                    Y_test = y_test
        
                else:
                    X_train = np.concatenate((X_train , x_train), axis = 1 )
                    X_val = np.concatenate((X_val , x_val), axis = 1 )
                    X_test = np.concatenate((X_test , x_test), axis = 1 )
                    
                    Y_train = np.concatenate((Y_train , y_train) , axis = 1)
                    Y_val = np.concatenate((Y_val , y_val) , axis = 1)
                    Y_test = np.concatenate((Y_test , y_test) , axis = 1)

    return X_train , X_val , X_test , Y_train , Y_val , Y_test     

def listfiles_mono(path , ext):
        
    lstFiles = []
     
    #Lista con todos los ficheros del directorio:
    lstDir = os.walk(path)   #os.walk()Lista directorios y ficheros
     
     
    #Crea una lista de los ficheros que existen en el directorio y los incluye a la lista.
     
    for root, dirs, files in lstDir:
        for fichero in files:
            (nombreFichero, extension) = os.path.splitext(fichero)
            if(extension == ext):
                lstFiles.append(nombreFichero+extension)
                #print (nombreFichero+extension)
                             
    print ('LISTADO FINALIZADO')
    print "longitud de la lista = ", len(lstFiles)
    return lstFiles

def list_mono():
    
    mono_path = '/home/tfgcartas/code/prueba1_mlp/mono'
    filename = listfiles_mono(mono_path , '.txt') 

    os.chdir(mono_path)   
    for i in range(len(filename)):
    
        note = np.loadtxt(filename[i] , delimiter= ',')
        n,m = os.path.splitext(filename[i])
        name = int(n[4::])
    
        l_train = int(note.size * 0.6)
        l_val_test = int(note.size * 0.2)
        x_train = note[:l_train]
        x_val = note[l_train:(note.size-l_val_test)]
        x_test = note[(note.size-l_val_test):note.size]
        
        C_train = np.abs(librosa.cqt(x_train, sr=44100, hop_length=512, n_bins=252, bins_per_octave=36))
        C_val = np.abs(librosa.cqt(x_val, sr=44100, hop_length=512, n_bins=252, bins_per_octave=36))
        C_test = np.abs(librosa.cqt(x_test, sr=44100, hop_length=512, n_bins=252, bins_per_octave=36))
        
        cy_train = np.zeros((88, C_train.shape[1]))
        cy_val = np.zeros((88, C_val.shape[1]))
        cy_test = np.zeros((88, C_test.shape[1]))
    
        cy_train[name-21, : ] = 1
        cy_val[name-21, : ] = 1
        cy_test[name-21, : ] = 1

        print(i)
        
        if(i<1):
            X_train = C_train
            X_val = C_val
            X_test = C_test
            
            y_train = cy_train
            y_val = cy_val
            y_test = cy_test
    
        else:
            X_train = np.concatenate((X_train , C_train), axis = 1 )
            X_val = np.concatenate((X_val , C_val), axis = 1 )
            X_test = np.concatenate((X_test , C_test), axis = 1 )
            
            y_train = np.concatenate((y_train , cy_train) , axis = 1)
            y_val = np.concatenate((y_val , cy_val) , axis = 1)
            y_test = np.concatenate((y_test , cy_test) , axis = 1)
    
    return X_train , X_val , X_test , y_train , y_val , y_test     



# define baseline model
def create_model(mainpath , num_classes):
    # create model
    model = Sequential()
    model.add(Dense(500, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(250, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(250, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    score_epoch_sampling = 3
    val_scores_list, val_loss_list, train_loss_list = [], [], []
    epc, best_val_score, patience_count = 0, 0, 0  # leave untouched

    nb_epoch = 200
    batch_size = 256
    patience = 15
    fig_path = mainpath + 'figures_png/' + 'figure_DNN_500_250_250_MP.png'

    keras_weight_file = mainpath + 'weights/' + 'weights_DNN_500_250_250_MP.h5'

    # Validation and patience
    while epc < nb_epoch:
        hist = model.fit(X_train, Y_train.T, epochs=1, batch_size=batch_size, validation_data=(X_val, Y_val.T), verbose=2)
        val_loss_list.append(hist.history.get('val_loss'))
        train_loss_list.append(hist.history.get('loss'))

        # Validation and patience
        if not epc % score_epoch_sampling:
            # Select samples from the validation set at random so we can visualize errors
            preds = model.predict(X_val, verbose=0)
            val_scores = dict()
            val_scores['f1_framewise'] = np.mean(f1_framewise(1.0 * (np.squeeze(preds) > 0.5), Y_val.T))
            val_scores['error_tot'] = error_tot(1.0 * (np.squeeze(preds) > 0.5), Y_val.T)
            val_scores_list.append(val_scores)
            print("* * validation scores:     ", val_scores)

            # save figure with losses and scores
            save_scores_in_fig(val_loss_list, train_loss_list, val_scores_list, fig_path)
            if val_scores['f1_framewise'] > best_val_score:
                best_val_score, patience_count = val_scores['f1_framewise'], 0
                model.save_weights(keras_weight_file, overwrite=True)
            else:
                patience_count += score_epoch_sampling

                if patience_count > patience:
                    print("Stopping training, validation score not improving after", best_val_score)
                    model.load_weights(keras_weight_file)
                    break
        epc += 1
        print('epc : ', epc, '    best_val_score  : ', best_val_score)

    model.load_weights(keras_weight_file)
    print("MODEL SAVED ", keras_weight_file)



    return model


def evaluation(X_test , Y_test , model):

    preds = model.predict(X_test, verbose=0)
    test_scores = dict()
    test_scores['f1_framewise'] = np.mean(f1_framewise(1.0 * (np.squeeze(preds) > 0.5), Y_test))
    test_scores['error_tot'] = error_tot(1.0 * (np.squeeze(preds) > 0.5), Y_test)
    os.chdir(mainpath + 'weights/')
    f1 = test_scores['f1_framewise']
    print("* * Test scores:     ", test_scores)
    print(f1)
    np.savetxt('test_scores_DNN_500_250_250_MP.txt' , f1 )

    return

##############################################################
####################### MAIN #################################


X_train_p , X_val_p , X_test_p , Y_train_p , Y_val_p , Y_test_p = list_poly() 
X_train_m , X_val_m , X_test_m , Y_train_m , Y_val_m , Y_test_m = list_mono() 


X_train = np.concatenate((X_train_p , X_train_m), axis = 1 )
X_val = np.concatenate((X_val_p , X_val_m), axis = 1 )
X_test = np.concatenate((X_test_p , X_test_m), axis = 1 )

Y_train = np.concatenate((Y_train_p , Y_train_m) , axis = 1)
Y_val = np.concatenate((Y_val_p , Y_val_m) , axis = 1)
Y_test = np.concatenate((Y_test_p , Y_test_m) , axis = 1)
    

########################################
print(X_train.shape , X_val.shape , X_test.shape)
print(Y_train.shape , Y_val.shape , Y_test.shape)

#Normalize
acumular(X_train , X_val)

X_train = normalizar(X_train)
X_val = normalizar(X_val)
X_test = normalizar(X_test)


print(np.mean(X_train),np.mean(X_val),np.mean(X_test))
print(np.std(X_train),np.std(X_val),np.std(X_test))
print(X_train.shape , X_val.shape , X_test.shape)
print(Y_train.shape , Y_val.shape , Y_test.shape)

########################################
num_classes = Y_test.shape[0]
print(num_classes)
mainpath = '/home/tfgcartas/code/mono+poly/'
# build the model
model = create_model(mainpath , num_classes)
#Final evaluation of the model
evaluation(X_test, Y_test.T , model)

scores = model.evaluate(X_test, Y_test.T, verbose=0)

print("Baseline Error: %.2f%%" % (100-scores[1]*100))
print(scores)

