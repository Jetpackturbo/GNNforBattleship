#!/usr/bin/env python
# coding: utf-8

import numpy as np

def comb_sum_product(phi_neg, phi_pos, theta):
    """" Perform sum-product in a comb-structured graph. More specifically,
    subgraph A, as pictured in figure 8.2(b) of the homework. The arguments
    are the unary log-potentials phi_neg and phi_pos, where phi_neg(i, j) is
    the log value of the potential if x(i, j) is -1, and phi_pos(i, j) is the
    log value if x(i, j) is 1. Theta is the coupling parameter defined in the
    problem. All unary potentials which lie outside the comb structure are
    ignored.

    The return values are the sum-product messages. m_left_neg(i, j) is the
    log value of the message from left neighbor x(i, j-1) to x(i, j) when
    x(i, j) takes the value -1. The other return values are defined
    analogously. All messages which do not correspond to edges in the comb
    graph are set to zero."""

    (M, N) = phi_neg.shape;
    m_up_neg = np.zeros((M, N))
    m_up_pos = np.zeros((M, N))
    m_down_neg = np.zeros((M, N))
    m_down_pos = np.zeros((M, N))
    m_left_neg = np.zeros((M, N))
    m_left_pos = np.zeros((M, N))
    m_right_neg = np.zeros((M, N))
    m_right_pos = np.zeros((M, N))    
    
    # up the columns
    for j in range(0,N,2):
        (m_neg, m_pos) = forward_messages_chain(phi_neg[-2::-1, j], phi_pos[-2::-1, j], theta)
        m_up_neg[-2::-1, j] = m_neg
        m_up_pos[-2::-1, j] = m_pos    

    # upwards messages become unary potentials in the chain model
    phi_neg_top = phi_neg[0, :].copy()
    phi_pos_top = phi_pos[0, :].copy()
    for j in range(0,N,2):
        phi_neg_top[j] = phi_neg_top[j] + m_up_neg[0, j]
        phi_pos_top[j] = phi_pos_top[j] + m_up_pos[0, j]

    # rightwards
    (m_neg, m_pos) = forward_messages_chain(phi_neg_top, phi_pos_top, theta)
    m_right_neg[0, :] = m_neg;
    m_right_pos[0, :] = m_pos;    
    
    # leftwards
    (m_neg, m_pos) = forward_messages_chain(phi_neg_top[::-1], phi_pos_top[::-1], theta)
    m_left_neg[0, ::-1] = m_neg
    m_left_pos[0, ::-1] = m_pos            
    
    # downwards    
    for j in range(0,N,2):
        curr_phi_neg = phi_neg[:-1, j]
        curr_phi_pos = phi_pos[:-1, j]
  
        # leftwards and rightward messages become unary potentials in the chain model
        curr_phi_neg[0] = curr_phi_neg[0] + m_right_neg[0, j]
        curr_phi_pos[0] = curr_phi_pos[0] + m_right_pos[0, j]
        curr_phi_neg[0] = curr_phi_neg[0] + m_left_neg[0, j]
        curr_phi_pos[0] = curr_phi_pos[0] + m_left_pos[0, j]
          
        (m_neg, m_pos) = forward_messages_chain(curr_phi_neg, curr_phi_pos, theta)
        m_down_neg[:-1, j] = m_neg
        m_down_pos[:-1, j] = m_pos

    return m_up_neg, m_up_pos, m_down_neg, m_down_pos, m_left_neg, m_left_pos, m_right_neg, m_right_pos

def forward_messages_chain(phi_neg, phi_pos, theta):
    N = phi_pos.size
    m_right_neg = np.zeros(N)
    m_right_pos = np.zeros(N)

    for i in range(1,N):
        m_right_neg[i] = logaddexp(m_right_neg[i-1] + phi_neg[i-1] + theta, m_right_pos[i-1] + phi_pos[i-1] - theta)
        m_right_pos[i] = logaddexp(m_right_neg[i-1] + phi_neg[i-1] - theta, m_right_pos[i-1] + phi_pos[i-1] + theta)
    
    return m_right_neg, m_right_pos

def logaddexp(a, b):
    mx = max(a, b)
    return np.log(np.exp(a - mx) + np.exp(b - mx)) + mx