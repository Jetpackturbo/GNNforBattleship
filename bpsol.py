#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

betavals = np.linspace(0.1, 3, 30)
l = 10
E = 4*l*(l-1)
Tmaxvals = np.array([15, 25])
delta = np.zeros((betavals.size, Tmaxvals.size))                 
# bhorfixed = np.random.uniform(size=(l, l-1))-0.5
# bvertfixed = np.random.uniform(size=(l-1, l))-0.5
# bnodesfixed = np.random.uniform(size=(l, l))-0.5
bhorfixed = np.random.uniform(size=(l, l-1))
bvertfixed = np.random.uniform(size=(l-1, l))
bnodesfixed = np.random.uniform(size=(l, l))

for iter1 in range(betavals.size):    
    Tmax = Tmaxvals.max()
    beta = betavals[iter1]
    
    # uniform random initialization of theta's in [0,beta]
    bhor = bhorfixed*beta
    bvert = bvertfixed*beta
    bnodes = bnodesfixed*beta
    
    # The following four matrices contain the messages (in log-likelihood format)
    # such that hhorright[i,j] = (1/2)log(nu_{(i,j)->(i,j+1)}(+1)/nu_{(i,j)->(i,j+1)}(-1))
    hhorright = np.zeros((l, l-1))  # hhorright[i,j] is the message from (i,j) to (i,j+1)
    hhorleft  = np.zeros((l, l-1))  # hhorleft[i,j] is the message from (i,j+1) to (i,j)
    hvertup   = np.zeros((l-1, l))  # hvertup[i,j] is the message from (i+1,j) to (i,j)
    hvertdown = np.zeros((l-1, l))  # hvertdown[i,j] is the message from (i,j) to (i+1,j)
    
    # The following matrices store the updated messages,
    hhorrightnew = np.zeros((l, l-1))
    hhorleftnew  = np.zeros((l, l-1))
    hvertupnew   = np.zeros((l-1, l))
    hvertdownnew = np.zeros((l-1, l))
    
    # for each instance of Tmax
    for t in range(Tmax):
        
        # looping over edges with 3 inputs
        for i in range(1,l-1):
            for j in range(1, l-1):                
                hhorrightnew[i, j] = bnodes[i, j] + np.sum(np.arctanh(np.array([np.tanh(bhor[i,j-1])*np.tanh(hhorright[i, j-1]), np.tanh(bvert[i-1,j])*np.tanh(hvertdown[i-1, j]), np.tanh(bvert[i, j])*np.tanh(hvertup[i, j])])))
                hvertdownnew[i, j] = bnodes[i, j] + np.sum(np.arctanh(np.array([np.tanh(bvert[i-1,j])*np.tanh(hvertdown[i-1, j]), np.tanh(bhor[i, j-1])*np.tanh(hhorright[i, j-1]), np.tanh(bhor[i, j])*np.tanh(hhorleft[i,j])])))

        for i in range(1,l-1):
            for j in range(l-2):
                hhorleftnew[i, j] = bnodes[i, j+1] + np.sum(np.arctanh(np.array([np.tanh(bhor[i, j+1])*np.tanh(hhorleft[i, j+1]), np.tanh(bvert[i-1, j+1])*np.tanh(hvertdown[i-1, j+1]), np.tanh(bvert[i, j+1])*np.tanh(hvertup[i, j+1])])))
                hvertupnew[j, i]  = bnodes[j+1, i] + np.sum(np.arctanh(np.array([np.tanh(bvert[j+1, i])*np.tanh(hvertup[j+1, i]), np.tanh(bhor[j+1, i-1])*np.tanh(hhorright[j+1, i-1]), np.tanh(bhor[j+1, i])*np.tanh(hhorleft[j+1, i])])))
        
        # looping over edges with 2 inputs
        for i in range(1,l-1):
            hvertupnew  [-1,  i] = bnodes[-1, i] + np.sum(np.arctanh(np.array([np.tanh(bhor [ -1, i-1])*np.tanh(hhorright[ -1, i-1]), np.tanh(bhor [-1,  i])*np.tanh(hhorleft[-1,  i])])))
            hvertdownnew[ 0,  i] = bnodes[ 0, i] + np.sum(np.arctanh(np.array([np.tanh(bhor [  0, i-1])*np.tanh(hhorright[  0, i-1]), np.tanh(bhor [ 0,  i])*np.tanh(hhorleft[ 0,  i])])))
            hhorrightnew[ i,  0] = bnodes[ i, 0] + np.sum(np.arctanh(np.array([np.tanh(bvert[i-1,   0])*np.tanh(hvertdown[i-1,   0]), np.tanh(bvert[ i,  0])*np.tanh(hvertup [ i,  0])])))
            hhorleftnew [ i, -1] = bnodes[ i,-1] + np.sum(np.arctanh(np.array([np.tanh(bvert[i-1,  -1])*np.tanh(hvertdown[i-1,  -1]), np.tanh(bvert[ i, -1])*np.tanh(hvertup [ i, -1])])))
            
        for i in range(1,l-1):
            hvertdownnew[ i,  0] = bnodes[ i, 0] + np.sum(np.arctanh(np.array([np.tanh(bvert[i-1,   0])*np.tanh(hvertdown[i-1,   0]), np.tanh(bhor [ i,  0])*np.tanh(hhorleft [  i,  0])])))
            hvertdownnew[ i, -1] = bnodes[ i,-1] + np.sum(np.arctanh(np.array([np.tanh(bvert[i-1,  -1])*np.tanh(hvertdown[i-1,  -1]), np.tanh(bhor [ i, -1])*np.tanh(hhorright[i-1, -1])])))
            hhorrightnew[ 0,  i] = bnodes[ 0, i] + np.sum(np.arctanh(np.array([np.tanh(bhor [  0, i-1])*np.tanh(hhorright[  0, i-1]), np.tanh(bvert[ 0,  i])*np.tanh(hvertup  [  0,  i])])))
            hhorrightnew[-1,  i] = bnodes[-1, i] + np.sum(np.arctanh(np.array([np.tanh(bhor [ -1, i-1])*np.tanh(hhorright[ -1, i-1]), np.tanh(bvert[-1,  i])*np.tanh(hvertdown[ -1,  i])])))
        
        for i in range(l-2):
            hvertupnew [ i,  0] = bnodes[i+1,  0] + np.sum(np.arctanh(np.array([np.tanh(bvert[i+1,   0])*np.tanh(hvertup [i+1,   0]), np.tanh(bhor [i+1,   0])*np.tanh(hhorleft [i+1,   0])])))
            hvertupnew [ i, -1] = bnodes[i+1, -1] + np.sum(np.arctanh(np.array([np.tanh(bvert[i+1,  -1])*np.tanh(hvertup [i+1,  -1]), np.tanh(bhor [i+1,  -1])*np.tanh(hhorright[i+1,  -1])])))
            hhorleftnew[ 0,  i] = bnodes[ 0, i+1] + np.sum(np.arctanh(np.array([np.tanh(bhor [  0, i+1])*np.tanh(hhorleft[  0, i+1]), np.tanh(bvert[  0, i+1])*np.tanh(hvertup  [  0, i+1])])))
            hhorleftnew[-1,  i] = bnodes[-1, i+1] + np.sum(np.arctanh(np.array([np.tanh(bhor [ -1, i+1])*np.tanh(hhorleft[ -1, i+1]), np.tanh(bvert[ -1, i+1])*np.tanh(hvertdown[ -1, i+1])])))
        
        # Edges with single input
        hvertdownnew[  0,  0] = bnodes[ 0,  0] + np.arctanh(np.tanh(bhor[0, 0])*np.tanh(hhorleft[0, 0]))
        hvertdownnew[  0, -1] = bnodes[ 0, -1] + np.arctanh(np.tanh(bhor[-2, 0]))
        hvertupnew  [ -1,  0] = bnodes[-1,  0] + np.arctanh(np.tanh(bhor[-1, 0])*np.tanh(hhorleft  [-1,  0]))
        hvertupnew  [ -1, -1] = bnodes[-1, -1] + np.arctanh(np.tanh(bhor[-1, -1])*np.tanh(hhorright[-1, -1]))
        
        hhorrightnew[ 0,  0] = bnodes[ 0,  0] + np.arctanh(np.tanh(bvert[ 0,  0])*np.tanh(hvertup  [ 0,  0]))
        hhorrightnew[-1,  0] = bnodes[-1,  0] + np.arctanh(np.tanh(bvert[-1,  0])*np.tanh(hvertdown[-1, -1]))
        hhorleftnew[  0, -1] = bnodes[ 0, -1] + np.arctanh(np.tanh(bvert[ 0, -2])*np.tanh(hvertup  [ 0, -2]))
        hhorleftnew[ -1, -1] = bnodes[-1, -1] + np.arctanh(np.tanh(bvert[-1, -1])*np.tanh(hvertdown[-1, -1]))
               
        # convert from log-likelihoods to probabilities
        nuvertdownnew = np.exp(2*hvertdownnew)/(1+np.exp(2*hvertdownnew))
        nuvertupnew   = np.exp(2*hvertupnew)  /(1+np.exp(2*hvertupnew))
        nuhorrightnew = np.exp(2*hhorrightnew)/(1+np.exp(2*hhorrightnew))
        nuhorleftnew  = np.exp(2*hhorleftnew) /(1+np.exp(2*hhorleftnew))
        
        nuvertdown = np.exp(2*hvertdown)/(1+np.exp(2*hvertdown))
        nuvertup   = np.exp(2*hvertup)  /(1+np.exp(2*hvertup))
        nuhorright = np.exp(2*hhorright)/(1+np.exp(2*hhorright))
        nuhorleft  = np.exp(2*hhorleft) /(1+np.exp(2*hhorleft))                
        
        # store the new messages as current messages
        hvertdown = hvertdownnew.copy()
        hvertup   = hvertupnew.copy()
        hhorleft  = hhorleftnew.copy()
        hhorright = hhorrightnew.copy() 
        
        # compute the difference delta
        if any(t+1==Tmaxvals):
            iter2 = np.where(t+1==Tmaxvals)
            delta[iter1, iter2] = (np.sum(abs(nuvertdownnew-nuvertdown)) + np.sum(abs(nuvertupnew-nuvertup)) + np.sum(abs(nuhorrightnew-nuhorright)) + np.sum(abs(nuhorleftnew-nuhorleft)))/E        

print(delta)

# plot resulting delta
plt.semilogy(betavals, delta,)
fig = plt.gcf()
fig.set_size_inches(fig.get_size_inches()*2)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\Delta(t)$')

plt.title("part c")
plt.legend([f"Tmax={Tmaxvals[0]}", f"Tmax={Tmaxvals[1]}"])
plt.grid()

plt.show()