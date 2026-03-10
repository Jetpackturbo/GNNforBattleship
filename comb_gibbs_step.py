#!/usr/bin/env python
# coding: utf-8

import numpy as np
from comb_sum_product import comb_sum_product

def comb_gibbs_step(x, theta):
    x = comb_gibbs_helper(x, theta)
    temp = comb_gibbs_helper(x[::-1, ::-1], theta)
    return temp[::-1, ::-1]

def comb_gibbs_helper(x, theta):
    (M, N) = x.shape
    phi_pos = np.zeros((M, N))
    
    # compute unary potentials in the tree graph
    for j in range(0,N-1,2):
        phi_pos[1:-1, j] = phi_pos[1:-1, j] + theta * x[1:-1, j+1]    
    for j in range(2,N,2):
        phi_pos[1:-1, j] = phi_pos[1:-1, j] + theta * x[1:-1, j-1]
    for j in range(1,N,2):
        phi_pos[0, j] = phi_pos[0, j] + theta * x[1, j]    
    for j in range(0,N,2):
        phi_pos[-2, j] = phi_pos[-2, j] + theta * x[-1, j]
       
    # sum product messages
    (m_up_neg, m_up_pos, m_down_neg, m_down_pos, m_left_neg, m_left_pos, m_right_neg, m_right_pos) =     comb_sum_product(-phi_pos.copy(), phi_pos.copy(), theta)
        
    # sample up the leftmost column
    x_new = x.copy()
    for i in range(M-2,0,-1):        
        odds = 2 * phi_pos[i, 0] + m_down_pos[i, 0] - m_down_neg[i, 0]
        if i < M-2:
            odds = odds + 2 * theta * x[i+1, 0]      
        x_new[i, 0] = sample_from_odds(odds)

    # sample right in the top row
    for j in range(N):        
        odds = 2 * phi_pos[0, j] + m_left_pos[0, j] - m_left_neg[0, j]        
        if j == 0:
            odds = odds + 2 * theta * x[1, 0]
        elif j % 2 == 0:
            odds = odds + m_up_pos[0, j] - m_up_neg[0, j]      

        if j > 0:
            odds = odds + 2 * theta * x[0, j-1]      

        x_new[0, j] = sample_from_odds(odds)    

    # sample down the remaining columns
    for j in range(2,N,2):
        for i in range(1,M-1):
            odds = 2 * phi_pos[i, j] + m_up_pos[i, j] - m_up_neg[i, j] + 2 * theta * x[i-1, j]
            x_new[i, j] = sample_from_odds(odds)
          
    return x_new

def sample_from_odds(odds):    
    prob = 1 / (1 + np.exp(-odds))
    return np.random.choice([1, -1], p=[prob, 1-prob])