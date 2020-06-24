# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from multiprocessing import Process, Value, Array, Manager
import numpy as np

#%%

def mark_points(im, N ,dictionary ,key):
    plt.figure()
    plt.imshow(im)
    plt.title(f'im{key}')
    p = plt.ginput(N, timeout = 0)   
    dictionary[key] = p 
   

def getPoints_par(im1: np.ndarray, im2: np.ndarray, N: int):
    manager = Manager()

    d = manager.dict()
    
    p1 = Process(target=mark_points, args=(im1, N, d, 1))
    p2 = Process(target=mark_points, args=(im2, N, d, 2))
    
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    return d[1],d[2]

def getPoints_seq(im1: np.ndarray, im2: np.ndarray, N: int):
    manager = Manager()

    d = manager.dict()
    
    mark_points(im1, N, d, 1)
    mark_points(im2, N, d, 2)
    
    return d[1],d[2]

#%%
def compute_H(p1, p2):
    N = len(p1)
    A = np.zeros((N*2, 9))
    A[::2, 2] = -1
    A[1::2, 5] = -1
    A[::2, :2] = -np.array(p2)
    A[1::2, 3:5] = -np.array(p2)
    p21 = np.concatenate((p2, np.ones((N, 1))), 1)
    p12 = p1.reshape(N,2,1)@p21.reshape(N,1, 3)
    A[:, 6:] = p12.reshape(2*N, 3)
    _, S, VT = np.linalg.svd(A)
    h = VT[-1]
    H2to1 = h.reshape(3, 3)
    return H2to1


def poject(p2, H):
    N = len(p2)
    p2_hgen = np.concatenate((p2, np.ones((N, 1))), 1)
    p1_computed = H @ p2_hgen.reshape(N, 3, 1)
    p1_cmp_nrm = np.round(p1_computed / p1_computed[:,2].reshape(N, 1, 1))
    p1_cmp_nrm = p1_cmp_nrm[:,:2].reshape(N, 2).astype(int)
    return p1_cmp_nrm
    
def get_points(im, N):
    plt.figure()
    plt.imshow(im)
    plt.title(f'get points')
    p = plt.ginput(N, timeout = 0) 
    return p
    
#%%

if __name__ == '__main__':
    
    plt.close('all')

    im1 = plt.imread('data/incline_L.png')
    im2 = plt.imread('data/incline_R.png')
    
    N = 5
    
    p1, p2 = getPoints_seq(im1, im2, N)
    p1, p2 = np.array(p1), np.array(p2)
    H = compute_H(p1, p2)    
    
    p3 = np.array(get_points(im2, 10))
    
    plt.figure()
    plt.imshow(im2)
    plt.scatter(p3[:,0], p3[:,1])
    
    p1_cmp_nrm = poject(p3, H) 
    
    plt.figure()
    plt.imshow(im1)
    plt.scatter(p1_cmp_nrm[:,0], p1_cmp_nrm[:,1])
    










































