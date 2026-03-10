#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def pair(x1,x2):
    """define pairwise potential function such that pair(x1,x2) = psi(x1,x2)"""
    if (x1==x2):
        out = 0.6
    else:
        out = 0.4
    return out

def single(x,y,epsilon,ForegroundMean,ForegroundVar,BackgroundMean,BackgroundVar):    
    """ define singleton potential function such that single(x,y) = phi(x,y)"""	
    if x==0:
        out = epsilon + (1/((2*np.pi)**(3/2)*np.sqrt(np.linalg.det(BackgroundVar))))*np.exp(-0.5*(y-BackgroundMean).dot(np.linalg.solve(BackgroundVar, y-BackgroundMean)))
    else:
        out = epsilon + (1/((2*np.pi)**(3/2)*np.sqrt(np.linalg.det(ForegroundVar))))*np.exp(-0.5*(y-ForegroundMean).dot(np.linalg.solve(ForegroundVar, y-ForegroundMean)))
    return out

def F(epsilon,FM,FC,BM,BC,y,MessageIn):
    """define BP update function that takes a set of messages as inputs
    and ouputs the updated message according to the BP update rule
    F(y,nu=[nu1 nu2 ...]) = log(phi(1,y)/phi(0,y)) + sum{log((psi(1,1)e^nu(i)+psi(1,0))/(psi(0,1)e^nu(i)+psi(0,0)))}"""

    A=[]
    for i in range(len(MessageIn)):
        A.append((pair(1,1)*np.exp(MessageIn[i])+pair(1,0))/(pair(0,1)*np.exp(MessageIn[i])+pair(0,0)))
    MessageOut = np.log(single(1,y,epsilon,FM,FC,BM,BC)/single(0,y,epsilon,FM,FC,BM,BC)) + sum(np.log(A))
    return MessageOut

# load image and compute the sample means and covariances
Im = (1/255)*plt.imread('flower.bmp').astype(np.double)
Foreground = plt.imread('foreground.bmp')
Background = plt.imread('background.bmp')
ImV = Im.transpose(2,1,0).reshape(3,-1)
Fo = Foreground.flatten('F')
Bo = Background.flatten('F')
ForegroundSample = ImV[:,Fo>0]
BackgroundSample = ImV[:,Bo>0]
FM = np.mean(ForegroundSample, 1)
FC = np.cov(ForegroundSample)
BM = np.mean(BackgroundSample, 1)
BC = np.cov(BackgroundSample)
print(f"Foreground mean: {FM}, Foreground covariance: {FC}")
print(f"Background mean: {BM}, Background covariance: {BC}")

epsilon = 0.01
niter = 30
nrow = Im.shape[0]
ncol = Im.shape[1]

# The following four matrices contain the messages (in log-likelihood format)
# such that hhorright(i,j) = log(nu_{(i,j)->(i,j+1)}(+1)/nu_{(i,j)->(i,j+1)}(0))
hhorright = np.zeros((nrow, ncol-1)) # hhorright(i,j) is the message from (i,j) to (i,j+1)
hhorleft  = np.zeros((nrow, ncol-1)) # hhorleft(i,j) is the message from (i,j+1) to (i,j)
hvertup   = np.zeros((nrow-1, ncol)) # hvertup(i,j) is the message from (i+1,j) to (i,j)
hvertdown = np.zeros((nrow-1, ncol)) # hvertdown(i,j) is the message from (i,j) to (i+1,j)

# The following matrices store the updated messages,
hhorrightnew = np.zeros((nrow, ncol-1))
hhorleftnew  = np.zeros((nrow, ncol-1))
hvertupnew   = np.zeros((nrow-1, ncol))
hvertdownnew = np.zeros((nrow-1, ncol))

# run parallel BP
for t in range(niter):
    # looping over edges with 3 inputs
    for i in range(1, nrow-1):
        for j in range(2, ncol-1):
            hhorrightnew[i  , j  ] = F(epsilon,FM,FC,BM,BC,Im[i,j,:],[hhorright[i  , j-1], hvertdown[i-1, j  ], hvertup [i, j]])
            hvertdownnew[i  , j  ] = F(epsilon,FM,FC,BM,BC,Im[i,j,:],[hvertdown[i-1, j  ], hhorright[i  , j-1], hhorleft[i, j]])
            hhorleftnew [i  , j-1] = F(epsilon,FM,FC,BM,BC,Im[i,j,:],[hhorleft [i  , j  ], hvertdown[i-1, j  ], hvertup [i, j]])
            hvertupnew  [i-1, j  ] = F(epsilon,FM,FC,BM,BC,Im[i,j,:],[hvertup  [i  , j  ], hhorright[i  , j-1], hhorleft[i, j]])

    # looping over edges with 2 inputs
    for i in range(1, nrow-1):
        hhorrightnew[i  , 0     ] = F(epsilon,FM,FC,BM,BC,Im[i, 0,:]     , [hvertdown[i-1, 0     ], hvertup  [i  ,0     ]])
        hhorleftnew [i  , ncol-2] = F(epsilon,FM,FC,BM,BC,Im[i, ncol-1,:], [hvertdown[i-1, ncol-1], hvertup  [i  ,ncol-1]])
        hvertdownnew[i  , 0     ] = F(epsilon,FM,FC,BM,BC,Im[i, 0,:]     , [hvertdown[i-1, 0     ], hhorleft [i  ,0     ]])
        hvertdownnew[i  , ncol-1] = F(epsilon,FM,FC,BM,BC,Im[i, ncol-1,:], [hvertdown[i-1, ncol-1], hhorright[i-1,ncol-2]])
        hvertupnew  [i-1, 0     ] = F(epsilon,FM,FC,BM,BC,Im[i, 0,:]     , [hvertup  [i  , 0     ], hhorleft [i  ,0     ]])
        hvertupnew  [i-1, ncol-1] = F(epsilon,FM,FC,BM,BC,Im[i, ncol-1,:], [hvertup  [i  , ncol-1], hhorright[i  ,ncol-2]])
    
    for i in range(1, ncol-1):
        hvertupnew  [nrow-2,   i] = F(epsilon,FM,FC,BM,BC,Im[nrow-1, i,:], [hhorright[nrow-1, i-1], hhorleft [nrow-1, i]])
        hvertdownnew[0     ,   i] = F(epsilon,FM,FC,BM,BC,Im[0     , i,:], [hhorright[0     , i-1], hhorleft [0     , i]])
        hhorrightnew[0     ,   i] = F(epsilon,FM,FC,BM,BC,Im[0     , i,:], [hhorright[0     , i-1], hvertup  [0     , i]])
        hhorrightnew[nrow-1,   i] = F(epsilon,FM,FC,BM,BC,Im[nrow-1, i,:], [hhorright[nrow-1, i-1], hvertdown[nrow-2, i]])
        hhorleftnew [0     , i-1] = F(epsilon,FM,FC,BM,BC,Im[0     , i,:], [hhorleft [0     , i  ], hvertup  [0     , i]])
        hhorleftnew [nrow-1, i-1] = F(epsilon,FM,FC,BM,BC,Im[nrow-1, i,:], [hhorleft [nrow-1, i  ], hvertdown[nrow-2, i]])    
    
    # Edges with single input
    hvertdownnew[0     , 0     ] = F(epsilon,FM,FC,BM,BC,Im[0     , 0,     :], [hhorleft [0     , 0     ]])
    hvertdownnew[0     , ncol-1] = F(epsilon,FM,FC,BM,BC,Im[0     , ncol-1,:], [hhorright[0     , ncol-2]])
    hvertupnew  [nrow-2, 0     ] = F(epsilon,FM,FC,BM,BC,Im[nrow-1, 0,     :], [hhorleft [nrow-1, 0     ]])
    hvertupnew  [nrow-2, ncol-1] = F(epsilon,FM,FC,BM,BC,Im[nrow-1, ncol-1,:], [hhorright[nrow-1, ncol-2]])
    
    hhorrightnew[0     , 0     ] = F(epsilon,FM,FC,BM,BC,Im[0     , 0,:     ], [hvertup  [0     , 0     ]])
    hhorrightnew[nrow-1, 0     ] = F(epsilon,FM,FC,BM,BC,Im[nrow-1, 0,:     ], [hvertdown[nrow-2, 0     ]])
    hhorleftnew [0     , ncol-2] = F(epsilon,FM,FC,BM,BC,Im[0     , ncol-1,:], [hvertup  [0     , ncol-2]])
    hhorleftnew [nrow-1, ncol-2] = F(epsilon,FM,FC,BM,BC,Im[nrow-1, ncol-1,:], [hvertdown[nrow-2, ncol-1]])
    
    # store the new messages as current messages
    hvertdown = hvertdownnew
    hvertup   = hvertupnew
    hhorleft  = hhorleftnew
    hhorright = hhorrightnew

# compute log likelihood ratio of the marginals
llr = np.zeros((nrow, ncol))

# nodes of degree 4
for i in range(1,nrow-1):
    for j in range(1, ncol-1):
        llr[i, j] = F(epsilon,FM,FC,BM,BC,Im[i,j,:], [hhorright[i, j-1], hvertdown[i-1, j], hvertup[i, j], hhorleft[i, j]])

# nodes of degree 3
for i in range(1, nrow-1):
    llr[i, 0     ] = F(epsilon,FM,FC,BM,BC,Im[i, 0     ,:], [hvertdown[i-1, 0     ], hvertup[i, 0     ], hhorleft [i, 0    ]])
    llr[i, ncol-1] = F(epsilon,FM,FC,BM,BC,Im[i, ncol-1,:], [hvertdown[i-1, ncol-1], hvertup[i, ncol-1], hhorright[i,ncol-2]])

for j in range(1, ncol-1):
    llr[0     ,j] = F(epsilon,FM,FC,BM,BC,Im[0     , j,:], [hhorright[0     , j-1], hhorleft[0     , j], hvertup  [0     , j]])
    llr[nrow-1,j] = F(epsilon,FM,FC,BM,BC,Im[nrow-1, j,:], [hhorright[nrow-1, j-1], hhorleft[nrow-1, j], hvertdown[nrow-2, j]])

# nodes of degree 2
llr[0     , 0     ] = F(epsilon,FM,FC,BM,BC,Im[0     , 0     ,:], [hhorleft [0     , 0     ], hvertup  [0     , 0     ]])
llr[0     , ncol-1] = F(epsilon,FM,FC,BM,BC,Im[0     , ncol-1,:], [hhorright[0     , ncol-2], hvertup  [0     , ncol-2]])
llr[nrow-1, 0     ] = F(epsilon,FM,FC,BM,BC,Im[nrow-1, 0     ,:], [hhorleft [nrow-1, 0     ], hvertdown[nrow-2, 0     ]])
llr[nrow-1, ncol-1] = F(epsilon,FM,FC,BM,BC,Im[nrow-1, ncol-1,:], [hhorright[nrow-1, ncol-2], hvertdown[nrow-2, ncol-1]])

# convert from log-likelihoods to probabilities
m = (np.exp(llr))/(1+np.exp(llr))
# plot the original image
# plt.figure()
# plt.imshow(Im)
# plot the expectation
plt.figure()
plt.title(f"BP with epsilon = {epsilon} and niter = {niter}, p(x_i = x_j) = 0.6")
plt.imshow(m)

plt.show()