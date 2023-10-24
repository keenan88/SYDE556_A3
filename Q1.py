# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:09:44 2023

@author: Keena
"""

import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from math import exp
    
def G_LIF_at_x(alpha, x, encoder, J_bias, Tref, Trc):
    
    J = alpha * x * encoder + J_bias
    
    if J > 1:
        G = 1 / (Tref - Trc * np.log(1 - 1/J))
    else:
        G = 0
        
    return G

def generate_LIF_tuning_curves(x_linspace, Tref, Trc, num_curves):
    
    tuning_curves = []
    alphas = []
    J_biases = []
    encoders = []
    
    
    for i in range(num_curves):
        
        a_max = np.random.uniform(100, 200)
        x_int = np.random.uniform(x_linspace[0], x_linspace[-1])
        encoder = np.random.choice(np.array([-1, 1]))
        r = 2
        
        # Alpha, J_bias
        K = 1 / (1 - exp((Tref - 1/a_max)/Trc))
        alpha = (K - 1) / (r - np.dot(x_int, encoder))
        J_bias = 1 - alpha * np.dot(x_int, encoder)
        
        alphas.append(alpha)
        J_biases.append(J_bias)
        encoders.append(encoder)
        
        tuning_curve = []
        
        for x in x_linspace:
            a = G_LIF_at_x(alpha, x, encoder, J_bias, Tref, Trc)
#            if x == 0:
#                print(x, a, alpha, J_bias)
            tuning_curve.append(a)
        
        tuning_curves.append(np.array(tuning_curve))

    return tuning_curves, alphas, J_biases, encoders

def get_RMSE_matrix(mat1, mat2):
    return round(np.sqrt(np.mean(np.square(mat1 - mat2))), 3)


if __name__ == "__main__":
    
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(18945)

    Tref = 2 / 1000 # Converted to seconds
    Trc = 20 / 1000 # Converted to seconds
    
    N = 20
    S = 41
    x_linspace = np.linspace(-2, 2, S)
    
    # 1A) [DONE]
    print("1.3B)")
    
    tuning_curves = generate_LIF_tuning_curves(x_linspace, Tref, Trc, N)
    
    for tuning_curve in tuning_curves:
        plt.plot(x_linspace, tuning_curve)

    plt.title("1A) " + str(N) + " LIF Tuning Curves")
    plt.xlabel("Current")
    plt.ylabel("Firitng Rate (Hz)")
    plt.xlim([x_linspace[0], x_linspace[-1]])
    plt.grid()
    plt.show() 
    
    #1B)
    
    A = np.matrix(tuning_curves)
    ro = 0.1 * 200
    noise_matrix = np.random.normal(0, ro, A.shape)
    A += noise_matrix
    X = np.matrix(x_linspace)
    
    
    normalizer = N * ro * ro * np.eye(N)
    decoders = (np.linalg.inv(A * A.T + normalizer) * A * X.T).T
    
    x_hat = (decoders * A).T
    
    difference = (x_hat.T - x_linspace).T
    
    plt.plot(x_linspace, difference)
    plt.title("1B) Real Minus Reconstructeud Response to Stimulus")
    plt.xlabel("Stimulus")
    plt.ylabel("Real Stimulus - Reconstructed Stimulus")
    plt.grid()
    plt.show()
    
    rmse = np.sqrt(np.mean(np.square(difference)))
    
    print("RMSE: ", rmse)
    

    














