# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:56:46 2019

@author: Francisco A
"""

def align(mid_file,Y_pred):

    import os
    import djitw
    import numpy as np
    import pretty_midi
    #import os 


    cwd = os.getcwd()
    print(cwd)
#    os.chdir('C:\Users\Francisco A\Google Drive (facm0002@red.ujaen.es)\ULI\TFG\Codigo primario\predicts+weigths\predicts_m+p')
    f = pretty_midi.PrettyMIDI(midi_file= mid_file)
    piano_roll = pretty_midi.PrettyMIDI.get_piano_roll(f , fs = 44100/512.0)
    piano_roll = piano_roll[21:109 , :]
    
    piano_roll_est = Y_pred
     
    for i in range(0 , piano_roll.shape[0]):
        for j in range(0 , piano_roll.shape[1]):
            if(piano_roll[i , j]> 0.5):
                piano_roll[i,j] = 1
    

    '''
    Align a MIDI object in-place to some audio data.
    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing some MIDI content
    audio_data : np.ndarray
        Samples of some audio data
    fs : int
        audio_data's sampling rate, and the sampling rate to use when
        synthesizing MIDI
    hop : int
        Hop length for CQT
    note_start : int
        Lowest MIDI note number for CQT
    n_notes : int
        Number of notes to include in the CQT
    penalty : float
        DTW non-diagonal move penalty
    '''
    # L2-normalized we can compute a cosine distance matrix via a dot product
    distance_matrix = 1 - np.dot(piano_roll[: , :].T, piano_roll_est[: , :])
    penalty = distance_matrix.mean()
    # Compute lowest-cost path through distance matrix
    p, q, score = djitw.dtw(distance_matrix, gully=.98, additive_penalty=penalty)
    
    return p, q

'''
p , q , score = align(piano_roll,piano_roll_est )


p = p.astype('int')
q = q.astype('int')

np.savetxt('p_alb_esp2_DNN500250250_x2.txt' , p.T , newline="," , fmt='%d')
np.savetxt('q_alb_esp2_DNN500250250_x2.txt' , q.T , newline="," , fmt='%d')

'''


