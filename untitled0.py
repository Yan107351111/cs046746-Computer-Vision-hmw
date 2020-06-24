# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 23:19:52 2020

@author: yan10
"""

        i = np.floor(nxs).astype(int)
        j = np.floor(nys).astype(int)
        
        px = (nxs-i)
        py = (nys-j)
        
        i_u = (i<=0).astype(bool)
        i_o = (i>self.xs[-1]-2).astype(bool)
        j_u = (j<=0).astype(bool)
        j_o = (j>self.ys[-1]-2).astype(bool)
        
        f00 = np.zeros_like(i)
        f00[1-(i_u+j_u)] = self.im[i[1-(i_u+j_u)]-1, j[1-(i_u+j_u)]-1]      #row0 col0 >> x0,y0
        
        f01 = np.zeros_like(i)
        f02 = np.zeros_like(i)
        f01[1-i_u] = self.im[i[1-i_u]-1, j[1-i_u]]      #row0 col1 >> x1,y0
        f02[1-i_u] = self.im[i[1-i_u], j[1-i_u]+1]      #row0 col2 >> x2,y0
        
        f10 = np.zeros_like(i)
        f20 = np.zeros_like(i)
        f10[1-j_u] = self.im[i[1-j_u]-1, j[1-j_u]]      #row0 col1 >> x1,y0
        f20[1-j_u] = self.im[i[1-j_u], j[1-j_u]+1]      #row0 col2 >> x2,y0
        
        f13 = np.zeros_like(i)
        f23 = np.zeros_like(i)
        f13[1-j_o] = self.im[i[1-j_o], j[1-j_o]+2]      #row0 col1 >> x1,y0
        f23[1-j_o] = self.im[i[1-j_o]+1, j[1-j_o]]+2      #row0 col1 >> x1,y0
        
        f31 = np.zeros_like(i)
        f32 = np.zeros_like(i)
        f31[1-i_o] = self.im[i[1-i_o]+2,j[1-i_o]]         #row3 col1 >> x1,y3
        f32[1-i_o] = self.im[i[1-i_o]+2,j[1-i_o]+1]       #row3 col2 >> x2,y3
        
        f03 = np.zeros_like(i)
        f30 = np.zeros_like(i)
        f33 = np.zeros_like(i)
        f03[(1-i_u)*(1-j_o)] = self.im[i[(1-i_u)*(1-j_o)]-1, j[(1-i_u)*(1-j_o)]+2]      #row0 col3 >> x3,y0
        f30[(1-i_o)*(1-j_u)] = self.im[i[(1-i_o)*(1-j_u)]-1, j[(1-i_o)*(1-j_u)]+2]      #row3 col0 >> x0,y3
        f33[(1-i_o)*(1-j_o)] = self.im[i[(1-i_o)*(1-j_o)]+2,j[(1-i_o)*(1-j_o)]+2]       #row3 col3 >> x3,y3
        
        
        
        f11 = p00 = self.im[i, j]    #row1 col1 >> x1,y1
        f12 = p01 = self.im[i, j+1]  #row1 col2 >> x2,y1

        f21 = p10 = self.im[i+1,j]   #row2 col1 >> x1,y2
        f22 = p11 = self.im[i+1,j+1] #row2 col2 >> x2,y2
        
        
        
        if 0 < i < self.xs[-1]-2 and 0 < j < self.ys[-1]-2:


            

        elif i<=0: 

            f03 = f02               #row0 col3 >> x3,y0

            f13 = f12               #row1 col3 >> x3,y1

            f23 = f22               #row2 col3 >> x3,y2

            f30 = self.im[i+2,j-1]       #row3 col0 >> x0,y3
            f31 = self.im[i+2,j]         #row3 col1 >> x1,y3
            f32 = self.im[i+2,j+1]       #row3 col2 >> x2,y3
            f33 = f32               #row3 col3 >> x3,y3             

        elif j<=0:

            f03 = self.im[i-1, j+2]      #row0 col3 >> x3,y0

            f13 = self.im[i,j+2]         #row1 col3 >> x3,y1

            f23 = self.im[i+1,j+2]       #row2 col3 >> x3,y2

            f30 = f20               #row3 col0 >> x0,y3
            f31 = f21               #row3 col1 >> x1,y3
            f32 = f22               #row3 col2 >> x2,y3
            f33 = f23               #row3 col3 >> x3,y3


        elif i == self.xs[-1]-2 or j == self.ys[-1]-2:

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
                     f30, f31, f32, f33]
        ).reshape(4,4).transpose() 
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
        
        Cr = Z@np.linalg.inv(X)
        R = Cr@np.stack([px**3, px**2, px, 1]).reshape(-1,4,1)
        
        Y = np.tile(np.array([-1, 0, 1, 2]), (4,1)).transpose()
        Y[:,0] = Y[:,0]**3
        Y[:,1] = Y[:,1]**2
        Y[:,-1] = 1
        
        Cc = np.linalg.inv(Y)@R
        z  = (Cc@np.array([py**3, py**2, py, 1]).reshape(-1,4,1)).reshape(-1)
        
        
        
        
        #%%
        
        
def warpH_v0(im1, H, out_size):
    """
    Your code here
    """
    
    x, y = np.meshgrid(np.arange(im1.shape[0]), np.arange(im1.shape[1]))
    p2 = np.stack((x.flatten(), y.flatten())).T
    p1_proj = poject(p2, H)
    ###
    # idxsx = np.arange(
    #     im1.shape[0]//2 - out_size[0]//2, 
    #     im1.shape[0] + np.ceil(out_size[0]/2)
    # )
    # idxsy = np.arange(
    #     im1.shape[1]//2 - out_size[1]//2, 
    #     im1.shape[1] + np.ceil(out_size[1]/2)
    # )
    # xx, yy = np.meshgrid(idxsx, idxsy)
    # p1_proj = np.stack((xx.flatten(), yy.flatten())).T
    ###
    p2_proj = poject(p1_proj, np.linalg.inv(H), dtype = float)
    accepted = (p2_proj[:, 0]>0)            \
              *(p2_proj[:, 0]<im1.shape[0]) \
              *(p2_proj[:, 1]>0)            \
              *(p2_proj[:, 1]<im1.shape[1])
    us = p2_proj[:, 0]
    vs = p2_proj[:, 1]
    xs = p1_proj[:, 0].astype(int)
    ys = p1_proj[:, 1].astype(int)
    print(xs, ys)
    x_rng = xs.max()+1
    y_rng = ys.max()+1
    warp_im1 = np.zeros((x_rng, y_rng, 3))
    
    for ch in range(3):
        interpolator = interp2d(
            np.arange(im1.shape[0]),
            np.arange(im1.shape[1]),
            im1[:,:,ch].flatten(),
            kind = 'linear'
        )
        bs = 100
        bn = len(p2_proj)//bs + 1
        for i in range(bn):
            warped_vals = interpolator(us[bs*i:bs*(i+1)], vs[bs*i:bs*(i+1)])
            bsip = len(warped_vals)
            # print('warped')
            # print(warped_vals)
            # print(warp_im1[xs[bs*i:bs*(i+1)], ys[bs*i:bs*(i+1)], ch])
            warp_im1[xs[bs*i:bs*(i+1)], ys[bs*i:bs*(i+1)], ch] = warped_vals[np.arange(bsip), np.arange(bsip)]
    return warp_im1


