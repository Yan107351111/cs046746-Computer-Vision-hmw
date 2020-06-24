import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt

#Add imports if needed:
from multiprocessing import Process, Value, Array, Manager
from scipy.interpolate import interp2d, SmoothBivariateSpline
from tqdm import tqdm
#end imports

#Add extra functions here:
def hgen2cart(p, dtype = int):
    if dtype is int:
        cart_p = np.round(p / p[:,2].reshape(-1, 1, 1))[:,:2].reshape(-1, 2).astype(dtype)
        return cart_p
    else:
        return (p / p[:,2].reshape(-1, 1, 1))[:,:2].reshape(-1, 2).astype(dtype)

def poject(p2, H, dtype = int):
    N = len(p2)
    p2_hgen = np.concatenate((p2, np.ones((N, 1))), 1)
    p1_computed = H @ p2_hgen.reshape(N, 3, 1)
    p1_cmp_nrm = hgen2cart(p1_computed, dtype = dtype)
    return p1_cmp_nrm

class yaninterp2d:
    '''
    class used to performe interpolation in image plane
    '''
    def __init__(self, xs: np.ndarray, ys: np.ndarray, im: np.ndarray, kind: str = 'linear'):
        '''
        ctor for image interpolator class.
        x and y coordinate should be the coordinate ranges of a uniform
        rectangular grid.

        Parameters
        ----------
        xs : np.ndarray
            x coordinates of the image samples.
        ys : np.ndarray
            y coordinates of the image samples.
        im : np.ndarray
            The available samples in image plane.
            array of shape [x max, y max]
        kind : str, optional
            The method to use for interpolation. must be one of {linear, cubic}
            The default is 'linear'.

        Returns
        -------
        None.

        '''
        self.xs = xs
        self.ys = ys
        self.im = im
        self.kind = kind
        
    def __call__(self, nxs, nys,):
        out = np.zeros_like(nxs)
        if len(out)>0:
            if self.kind == 'linear':
                out = self.linterp(
                    nxs = nxs,
                    nys = nys,
                    bs = len(out)
                )
            if self.kind == 'cubic':
                out = self.cubterp(
                    nxs = nxs,
                    nys = nys,
                    bs = len(out)
                )
        return out
    
    def linterp(self, nxs, nys, bs):
        '''
        perform linear interpolation for values in new points

        Parameters
        ----------
        nxs : np.ndarray
            x values of new points.
        nys : np.ndarray
            y values of new points.
        bs : int
            batch size.

        Returns
        -------
        out : np.ndarray
            values in new points.

        '''
        im = np.zeros((self.im.shape[0]+1, self.im.shape[1]+1,))
        im[:-1, :-1] = self.im
        out = np.zeros((bs,))
        ongrid = (nxs==np.floor(nxs))*(nys==np.floor(nys)).astype(bool)
        # print(ongrid)
        out[ongrid] = self.im[nys.astype(int)[ongrid], nxs.astype(int)[ongrid]]
        offgrid = (1-ongrid).astype(bool)
        nxs = nxs[offgrid]
        nys = nys[offgrid]
        # print(f'nxs: {nxs}')
        # print(f'nys: {nys}')
        out[offgrid]  = im[np.floor(nys).astype(int), np.floor(nxs).astype(int)] \
                     *(np.ceil(nxs)-nxs)*(np.ceil(nys)-nys)
        out[offgrid] += im[np.floor(nys).astype(int), np.ceil(nxs).astype(int)]  \
                     *(np.ceil(nxs)-nxs)*(nys-np.floor(nys))
        out[offgrid] += im[np.ceil(nys).astype(int), np.floor(nxs).astype(int)]  \
                     *(nxs-np.floor(nxs))*(np.ceil(nys)-nys)
        out[offgrid] += im[np.ceil(nys).astype(int), np.ceil(nxs).astype(int)]  \
                     *(nxs-np.floor(nxs))*(nys-np.floor(nys))
        # print(out)
        return out
    #'''  
    def cubterp(self, nxs, nys, bs):
        '''
        perform cubic interpolation for values in new points
        code adapted from:
        https://stackoverflow.com/questions/52700878/bicubic-interpolation-python

        Parameters
        ----------
        nxs : np.ndarray
            x values of new points.
        nys : np.ndarray
            y values of new points.
        bs : int
            batch size.

        Returns
        -------
        out : np.ndarray
            values in new points.
            
        '''
        xi = self.xs
        yi = self.ys
        # zi = np.zeros((len(xi)+3, len(yi)+3))
        
        zi = self.im
        xnew = nxs
        ynew = nys
        # check sorting
        if np.any(np.diff(xi) < 0) and np.any(np.diff(yi) < 0) and\
        np.any(np.diff(xnew) < 0) and np.any(np.diff(ynew) < 0):
            raise ValueError('data are not sorted')
    
        # if zi.shape != (xi.size, yi.size):
        #     raise ValueError('zi is not set properly use np.meshgrid(xi, yi)')
    
        z = np.zeros((xnew.size,))
    
        deltax = xi[1] - xi[0]
        deltay = yi[1] - yi[0] 
        for n, (x, y) in enumerate(zip(xnew, ynew)):
            if xi.min() <= x <= xi.max() and yi.min() <= y <= yi.max():

                j = np.searchsorted(xi, x) - 1
                i = np.searchsorted(yi, y) - 1
                

                x1  = xi[j]
                x2  = xi[j+1]

                y1  = yi[i]
                y2  = yi[i+1]

                px = (x-x1)/(x2-x1)
                py = (y-y1)/(y2-y1)

                f00 = zi[i-1, j-1]      #row0 col0 >> x0,y0
                f01 = zi[i-1, j]        #row0 col1 >> x1,y0
                f02 = zi[i-1, j+1]      #row0 col2 >> x2,y0

                f10 = zi[i, j-1]        #row1 col0 >> x0,y1
                f11 = p00 = zi[i, j]    #row1 col1 >> x1,y1
                f12 = p01 = zi[i, j+1]  #row1 col2 >> x2,y1

                f20 = zi[i+1,j-1]       #row2 col0 >> x0,y2
                f21 = p10 = zi[i+1,j]   #row2 col1 >> x1,y2
                f22 = p11 = zi[i+1,j+1] #row2 col2 >> x2,y2

                if 0 < j < xi.size-2 and 0 < i < yi.size-2:

                    f03 = zi[i-1, j+2]      #row0 col3 >> x3,y0

                    f13 = zi[i,j+2]         #row1 col3 >> x3,y1

                    f23 = zi[i+1,j+2]       #row2 col3 >> x3,y2

                    f30 = zi[i+2,j-1]       #row3 col0 >> x0,y3
                    f31 = zi[i+2,j]         #row3 col1 >> x1,y3
                    f32 = zi[i+2,j+1]       #row3 col2 >> x2,y3
                    f33 = zi[i+2,j+2]       #row3 col3 >> x3,y3

                elif i<=0: 

                    f03 = f02               #row0 col3 >> x3,y0

                    f13 = f12               #row1 col3 >> x3,y1

                    f23 = f22               #row2 col3 >> x3,y2

                    f30 = zi[i+2,j-1]       #row3 col0 >> x0,y3
                    f31 = zi[i+2,j]         #row3 col1 >> x1,y3
                    f32 = zi[i+2,j+1]       #row3 col2 >> x2,y3
                    f33 = f32               #row3 col3 >> x3,y3             

                elif j<=0:

                    f03 = zi[i-1, j+2]      #row0 col3 >> x3,y0

                    f13 = zi[i,j+2]         #row1 col3 >> x3,y1

                    f23 = zi[i+1,j+2]       #row2 col3 >> x3,y2

                    f30 = f20               #row3 col0 >> x0,y3
                    f31 = f21               #row3 col1 >> x1,y3
                    f32 = f22               #row3 col2 >> x2,y3
                    f33 = f23               #row3 col3 >> x3,y3


                elif j == xi.size-2 or i == yi.size-2:

                    f03 = f02               #row0 col3 >> x3,y0

                    f13 = f12               #row1 col3 >> x3,y1

                    f23 = f22               #row2 col3 >> x3,y2

                    f30 = f20               #row3 col0 >> x0,y3
                    f31 = f21               #row3 col1 >> x1,y3
                    f32 = f22               #row3 col2 >> x2,y3
                    f33 = f23               #row3 col3 >> x3,y3

                Z = np.array([f00, f01, f02, f03,
                             f10, f11, f12, f13,
                             f20, f21, f22, f23,
                             f30, f31, f32, f33]).reshape(4,4).transpose()

                X = np.tile(np.array([-1, 0, 1, 2]), (4,1))
                X[0,:] = X[0,:]**3
                X[1,:] = X[1,:]**2
                X[-1,:] = 1

                Cr = Z@np.linalg.inv(X)
                R = Cr@np.array([px**3, px**2, px, 1])

                Y = np.tile(np.array([-1, 0, 1, 2]), (4,1)).transpose()
                Y[:,0] = Y[:,0]**3
                Y[:,1] = Y[:,1]**2
                Y[:,-1] = 1

                Cc = np.linalg.inv(Y)@R

                z[n]=(Cc@np.array([py**3, py**2, py, 1]))
        return z
    
    '''   
    def cubterp(self, nxs, nys, bs):
        '#''
        perform cubic interpolation for values in new points
        code adapted from:
        https://stackoverflow.com/questions/52700878/bicubic-interpolation-python

        Parameters
        ----------
        nxs : np.ndarray
            x values of new points.
        nys : np.ndarray
            y values of new points.
        bs : int
            batch size.

        Returns
        -------
        out : np.ndarray
            values in new points.
            
        '#''
        xi = self.xs
        yi = self.ys
        # print(f'self.xs: {self.xs.shape}')
        # print(f'self.ys {self.ys.shape}')
        pad = 3
        zi = np.zeros((len(yi)+pad, len(xi)+pad))
        # print(f'self.im.shape: {self.im.shape}')
        zi[pad//2:-1-pad//2, pad//2:-1-pad//2] = self.im
        
        i = np.floor(nxs).astype(int)+pad//2
        j = np.floor(nys).astype(int)+pad//2
        
        px = (nxs-i)+pad//2
        py = (nys-j)+pad//2
        
        
        f00 = zi[j-1, i-1]      #row0 col0 >> x0,y0
        f01 = zi[j-1, i]      #row0 col1 >> x1,y0
        f02 = zi[j, i+1]      #row0 col2 >> x2,y0
        f10 = zi[j-1, i]      #row0 col1 >> x1,y0
        f20 = zi[j, i+1]      #row0 col2 >> x2,y0
        f13 = zi[j, i+2]      #row0 col1 >> x1,y0
        f23 = zi[j+1, i]+2      #row0 col1 >> x1,y0
        f31 = zi[j+2,i]         #row3 col1 >> x1,y3
        f32 = zi[j+2,i+1]       #row3 col2 >> x2,y3
        f03 = zi[j-1, i+2]      #row0 col3 >> x3,y0
        f30 = zi[j-1, i+2]      #row3 col0 >> x0,y3
        f33 = zi[j+2,i+2]       #row3 col3 >> x3,y3
        f11 = zi[j, i]    #row1 col1 >> x1,y1
        f12 = zi[j, i+1]  #row1 col2 >> x2,y1
        f21 = zi[j+1,i]   #row2 col1 >> x1,y2
        f22 = zi[j+1,i+1] #row2 col2 >> x2,y2
        
        Z = np.stack(
            [f00, f01, f02, f03,
             f10, f11, f12, f13,
             f20, f21, f22, f23,
             f30, f31, f32, f33]
        ).reshape(-1,4,4).transpose(0,2,1)
        
        X = np.tile(np.array([-1, 0, 1, 2]), (4,1))
        X[0,:] = X[0,:]**3
        X[1,:] = X[1,:]**2
        X[-1,:] = 1
        
        Cr = Z@np.linalg.inv(X).reshape(1,4,4)
        print(f'Cr.shape: {Cr.shape}')
        R = Cr@np.stack([px**3, px**2, px, np.ones_like(px)]).reshape(-1,4,1)
        print(f'R.shape: {R.shape}')
        Y = np.tile(np.array([-1, 0, 1, 2]), (4,1)).transpose()
        Y[:,0] = Y[:,0]**3
        Y[:,1] = Y[:,1]**2
        Y[:,-1] = 1
        
        Cc = (np.linalg.inv(Y).reshape(1,4,4)@R).transpose(0,2,1)
        print(f'Cc.shape: {Cc.shape}')
        pyv = np.array([py**3, py**2, py, np.ones_like(py)]).reshape(-1,4,1)
        print(f'Cc: {Cc.shape}')
        print(f'pyv: {pyv.shape}')
        print(f'Cc@pyv: {(Cc@pyv).shape}')
        
        z  = (Cc@pyv).reshape(-1)
        # z  = (pyv@Cc).reshape(-1)
        return z
    '''

def create_pyrs(A,B):
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5,0,-1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5,0,-1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)
    return lpA,lpB

def blend_images(A,B):
    lpA,lpB = create_pyrs(A,B)
    # Now add left and right halves of images in each level
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):])) #mixing can also be done with a mask
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    # image with direct connecting each half
    real = np.hstack((A[:,:int(cols/2)],B[:,int(cols/2):])) 
    return real, ls_

def switch_texture(A,B):
    lpA,lpB = create_pyrs(A,B)
    # Now add left and right halves of images in each level
    LS = []
#     for la,lb in zip(lpA,lpB):
#         rows,cols,dpt = la.shape
#         ls = np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):])) #mixing can also be done with a mask
#         LS.append(ls)
    # now reconstruct
    ls_ = lpA[0]
    for i in range(1,6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, lpB[i])
    # image with direct connecting each half
    real = np.hstack((A[:,:int(cols/2)],B[:,int(cols/2):])) 
    return real, ls_

def alpha_blend(A,B,MASK):
    A = np.float32(A)
    B = np.float32(B)
    return np.uint8(A*MASK), np.uint8(A*MASK+B*(1-MASK))

def padalign_ims(wim1, im2, biases):
    y1, x1, _ = wim1.shape
    y2, x2, _ = im2.shape
    
    
    if biases[0]>=0:
        x_s = max(x2, x1 + biases[0]+1)
    if biases[0]<0:
        x_s = max(x1, x2 - biases[0]+1)
    if biases[1]>=0:
        y_s = max(y2, y1 + biases[1]+1)
    if biases[1]<0:
        y_s = max(y1, y2 - biases[1]+1)
        
    print(x_s, y_s)
    wim1_padded = np.zeros((y_s, x_s, 3))
    im2_padded = np.zeros((y_s, x_s, 3))
    
    if biases[0]>=0:
        if biases[1]>=0:
            wim1_padded[biases[1]:biases[1]+y1, biases[0]:biases[0]+x1, :] = wim1
            im2_padded[0:y2, 0:x2, :] = im2
        else:
            wim1_padded[biases[1]:biases[1]+y1, 0:x1, :] = wim1
            im2_padded[0:y2, -biases[0]:-biases[0]+x2, :] = im2
    else:
        if biases[0]>=0:
            wim1_padded[0:y1, biases[0]:biases[0]+x1, :] = wim1
            im2_padded[-biases[1]:-biases[1]+y2, 0:x2, :] = im2
        else:
            print(wim1.shape[1])
            print(wim1_padded[0:y1, 0:x1, :].shape)
            wim1_padded[0:y1, 0:x1, :] = wim1
            im2_padded[-biases[1]:-biases[1]+y2, -biases[0]:-biases[0]+x2, :] = im2

    return wim1_padded, im2_padded
#Extra functions end

# HW functions:


def mark_points(im, N, key):
    plt.figure()
    plt.imshow(im)
    plt.title(f'im{key}')
    p = plt.ginput(N, timeout = 0)
    return p
    
def getPoints(im1,im2,N):
    """
    Your code here
    """
    p1 = mark_points(im1, N, 1)
    p2 = mark_points(im2, N, 2)
    return p1,p2

def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    p1 = p1.T
    p2 = p2.T
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

def warpH(im1, H, out_size, get_biases = False):
    """
    Your code here
    """
    out_size = im1.shape
    
    
    y, x = np.meshgrid(np.arange(out_size[0]), np.arange(out_size[1]))
    # x += im1.shape[0]//2 - out_size[0]//2
    # y += im1.shape[1]//2 - out_size[1]//2
    # print(f'x, y: {x, y}')
    p1 = np.stack((x.flatten(), y.flatten())).T
    p2_proj = poject(p1, np.linalg.inv(H))
    p1_proj  = poject(p2_proj, H, dtype = float)
    # print(f'p1_proj: {p1_proj}')
    
    # plt.figure(100)
    # plt.subplot(1,3,1)
    # plt.imshow(im1)
    # plt.scatter(p1[::314,0], p1[::314,1],  s = 1)
    
    # # print(p1_cmp_nrm.shape)
    # # print(p1_cmp_nrm[1000:1100])
    # plt.subplot(1,3,2)
    # plt.imshow(im2)
    # plt.scatter(p2_proj[::314,0], p2_proj[::314,1], s = 1)
    
    # plt.subplot(1,3,3)
    # plt.imshow(im1)
    # plt.scatter(p1_proj[::314,0], p1_proj[::314,1], s = 1)
    
    accepted = (p1_proj[:, 0]>0)            \
              *(p1_proj[:, 0]<im1.shape[0]) \
              *(p1_proj[:, 1]>0)            \
              *(p1_proj[:, 1]<im1.shape[1])
    accepted = accepted.astype(bool)
    # print(accepted[500:550])
    us = p1_proj[:, 0]#[accepted]
    vs = p1_proj[:, 1]#[accepted]
    # print(f'us: {us}')
    # print(f'vs: {vs}')
    
    x_bias = p2_proj[:, 0].min()
    y_bias = p2_proj[:, 1].min()
    
    xs = (p2_proj[:, 0] - x_bias)#[accepted]
    ys = (p2_proj[:, 1] - y_bias)#[accepted]
    # print(f'xs: {xs}')
    # print(f'ys: {ys}')
    # warp_im1 = np.zeros((*out_size[:2], 3))
    
    # plt.figure(101)
    # plt.subplot(1,3,1)
    # plt.imshow(im1)
    # plt.scatter(us[::314], vs[::314])
    
    # # print(p1_cmp_nrm.shape)
    # # print(p1_cmp_nrm[1000:1100])
    # plt.subplot(1,3,2)
    # plt.imshow(im2)
    # plt.scatter(xs[::314], ys[::314], s = 1)
    
    # plt.subplot(1,3,3)
    # plt.imshow(im1)
    # plt.scatter(p1_proj[::314,0], p1_proj[::314,1], s = 1)
    
    x_rng = xs.max()+1
    y_rng = ys.max()+1
    warp_im1 = np.zeros((y_rng, x_rng, 3))
    # print(f'x_rng: {x_rng}')
    # print(f'y_rng: {y_rng}')
    for ch in range(3):
        interpolator = yaninterp2d(
            np.arange(im1.shape[1]),
            np.arange(im1.shape[0]),
            im1[:,:,ch],#.flatten(),
            kind = 'linear',
        )
        # print(f'warp_im1.shape: {warp_im1.shape}')
        warp_im1[ys, xs, ch] = interpolator(us, vs)
    
        '''
        bs = 100
        bn = len(p1_proj)//bs + 1
        # print(f'bn: {bn}')
        for i in tqdm(range(bn)):
            if len(us[bs*i:bs*(i+1)])==0:
                continue
            # print(f'us[bs*i:bs*(i+1)]: {us[bs*i:bs*(i+1)]}')
            # print(f'vs[bs*i:bs*(i+1)]: {vs[bs*i:bs*(i+1)]}')
            warped_vals = interpolator(us[bs*i:bs*(i+1)], vs[bs*i:bs*(i+1)])
            bsip = len(warped_vals)
            # print('warped')
            # print(f'warped_vals: {warped_vals}')
            # print(warp_im1[xs[bs*i:bs*(i+1)], ys[bs*i:bs*(i+1)], ch])
            # print(f'warp_im1.shape: {warp_im1.shape}')
            # print(f'warp_im1: {warp_im1[x[bs*i:bs*(i+1)], y[bs*i:bs*(i+1)], ch]}')
            # print(f'xs[bs*i:bs*(i+1)]: {xs[bs*i:bs*(i+1)]}')
            # print(f'ys[bs*i:bs*(i+1)]: {ys[bs*i:bs*(i+1)]}')
            warp_im1[ys[bs*i:bs*(i+1)], xs[bs*i:bs*(i+1)], ch] = warped_vals[np.arange(bsip), np.arange(bsip)]
    
        '''
        yy, xx = np.meshgrid(np.arange(1,warp_im1.shape[0]-1), np.arange(1,warp_im1.shape[1]-1))
        yy = yy.flatten()
        xx = xx.flatten()
        bb = (warp_im1[yy, xx, ch]==0)      \
            *((warp_im1[yy-1, xx-1, ch]!=0) \
             +(warp_im1[yy, xx-1, ch]!=0)   \
             +(warp_im1[yy-1, xx, ch]!=0)   \
             +(warp_im1[yy-1, xx+1, ch]!=0) \
             +(warp_im1[yy, xx+1, ch]!=0)   \
             +(warp_im1[yy+1, xx-1, ch]!=0) \
             +(warp_im1[yy+1, xx, ch]!=0)   \
             +(warp_im1[yy+1, xx+1, ch]!=0))
        bb = bb.astype(bool)
        # print(f'bb: {bb.shape}')
        # print(f'yy: {yy.shape}')
        # print(f'xx: {xx.shape}')
        
        yys = yy[bb]
        xxs = xx[bb]
        mp2 = np.stack((xxs.flatten()+x_bias, yys.flatten()+y_bias)).T
        
        mp1_proj = poject(mp2, H, dtype = float)
        
        
        accepted = (mp1_proj[:, 1]>0)            \
                  *(mp1_proj[:, 1]<im1.shape[0]-1) \
                  *(mp1_proj[:, 0]>0)            \
                  *(mp1_proj[:, 0]<im1.shape[1]-1)
        accepted = accepted.astype(bool)

        mus = mp1_proj[:, 0][accepted]
        mvs = mp1_proj[:, 1][accepted]
        
        # plt.figure(101)
        # plt.subplot(1,3,1)
        # plt.imshow(im1)
        # plt.scatter(mus[::2], mvs[::2], s = 1)
        
        # plt.subplot(1,3,2)
        # plt.imshow(im2)
        # plt.scatter(xxs[::2]+x_bias, yys[::2]+y_bias, s = 1)
        
        # plt.subplot(1,3,3)
        # plt.imshow(im1)
        # plt.scatter(mp1_proj[::2,0], mp1_proj[::2,1], s = 1)


        warp_im1[yys[accepted], xxs[accepted], ch] = interpolator(mus, mvs)
        #'''
    if get_biases:
        return warp_im1, [x_bias, y_bias]
    return warp_im1

def imageStitching(img1, wrap_img2):
    """
    Your code here
    """
    mask1 = np.sum(img1, 2)>0
    mask2 = np.sum(wrap_img2, 2)>0
    
    inter = mask1*mask2
    # interxs = np.min(inter*np.arange(img1.shape[1]).reshape(1, -1))
    # interxe = np.max(inter*np.arange(img1.shape[1]).reshape(1, -1))
    # interys = np.min(inter*np.arange(img1.shape[0]).reshape(-1, 1))
    # interye = np.max(inter*np.arange(img1.shape[0]).reshape(-1, 1))
    
    weight1 = np.ones_like(img1)
    # weight2 = np.ones_like(wrap_img2)
    
    # wxx , wyy = np.meshgrid(
    #     np.arange(interxs-interxe),
    #     np.arange(interys-interye),
    # )
    
    # weight1[interxs:interxe, interys:interye] = 1
    weight2 = (1-inter).reshape(*inter.shape, 1)
    # plt.figure()
    # plt.imshow(wrap_img2)
    # plt.figure()
    # plt.imshow(weight2)
    # plt.figure()
    # plt.imshow(mask1)
    # plt.figure()
    # plt.imshow(mask2)
    # plt.figure()
    # plt.imshow(inter)
    
    # weight1[interxs:interxe, interys:interye] = \
    #     1 - wxx/(interxs-interxe-1)*0.5 \
    #     - wyy/(interys-interye-1)*0.5
    
    # weight2[interxs:interxe, interys:interye] = \
    #     wxx/(interxs-interxe-1)*0.5 \
    #     + wyy/(interys-interye-1)*0.5 - 1

    panoImg = img1*weight1 + wrap_img2*weight2
    
    # mass1 = np.sum(mask1)
    # mass2 = np.sum(mask2)

    # cg1x = mask1*np.arange(img1.shape[1]).reshape(1, -1)/mass1
    # cg1y = mask1*np.arange(img1.shape[0]).reshape(-1, 1)/mass1
    # cg2x = mask2*np.arange(wrap_img2.shape[1]).reshape(1, -1)/mass2
    # cg2y = mask2*np.arange(wrap_img2.shape[0]).reshape(-1, 1)/mass2
    
    # cg1 = np.array([cg1x, cg1y])
    # cg2 = np.array([cg2x, cg2y])
    # trans_direc1 = cg2-cg1
    # trans_direc2 = cg1-cg2
    
    # weight1[inter.astype(bool)] = 1-
    # iom1 = np.sum(inter).astype(float)/mass1
    # iom2 = np.sum(inter).astype(float)/mass2
    # dim_start = 
    return panoImg

def ransacH(matches, locs1, locs2, nIter, tol):
    """
    Your code here
    """
    bestH = 0
    return bestH

def getPoints_SIFT(im1,im2):
    """
    Your code here
    """
    p1,p2 = 0, 0
    return p1,p2

if __name__ == '__main__':
    print('my_homography')
    im1 = plt.imread('data/incline_L.png')
    im2 = plt.imread('data/incline_R.png')

    N = 5
    #%%
    plt.close('all')
    # p1, p2 = getPoints(im1, im2, N)
    # p1, p2 = np.array(p1).T, np.array(p2).T
    # H = computeH(p1, p2)
    # p3 = np.array(mark_points(im2, 5, 2))
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im2)
    plt.scatter(p3[:,0], p3[:,1])
    
    p1_cmp_nrm = poject(p3, H) 
    plt.subplot(1,2,2)
    plt.imshow(im1)
    plt.scatter(p1_cmp_nrm[:,0], p1_cmp_nrm[:,1])
    
    
    plt.figure(10)
    plt.subplot(3,1,1)
    plt.imshow(im1)
    wim1 = warpH(im1, H, (10, 10))
    plt.figure(10)
    plt.subplot(3,1,2)
    plt.imshow(wim1)
    plt.subplot(3,1,3)
    plt.imshow(im2)

    
    wim1, biases = warpH(im1, H, 0, get_biases = True)
    
    wim1_padded, im2_padded = padalign_ims(wim1, im2, biases)
    im_stitch = imageStitching(im2_padded, wim1_padded)
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(wim1_padded)
    plt.subplot(2,1,2)
    plt.imshow(im2_padded)
    plt.figure()
    plt.imshow(im_stitch)
    
    """
    Your code here
    """
