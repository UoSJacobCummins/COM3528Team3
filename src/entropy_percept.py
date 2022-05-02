
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import entropy
from scipy import ndimage

# imo = cv2.imread('ball1.jpg')
# imo = cv2.imread('room_balls.jpg')

def computeEntropy( I, Q = None):
    # x,b = np.histogram( I, 20 )
    x = I.flatten()
    x = x/float(sum(x))

    if Q is not None:
        y = Q.flatten()
        # y,b = np.histogram( Q, 20 )
        y = y/float(sum(y))
        # print x
        # print y
        y[y == 0] = 1e-10
        x[x == 0] = 1e-10
        # b = (b[1:] + b[0:-1])/2.0
        h = entropy(x.astype(np.float32), y.astype(np.float32))
    else:
        h = entropy(x)
    return h

def getEntropy( im ):

    m,n = im.shape
    num_n = 40
    num_m = 40
    im = im/255.0
    sn = n//num_n
    sm = m//num_m
    alo = m//2
    alf = alo + sn
    aro = 330
    arf = aro + sm
    imr = np.zeros(im.shape)
    I = np.zeros((sm,sn))

    for i in range(num_m):
        for j in range(num_n):
            lo = sm*i
            lf = min(sm*(i+1),m)
            ro = sn*j
            rf = min(sn*(j+1),n)

            I += im[lo:lf,ro:rf]
            
    I /= num_n*num_m

    for i in range(num_m):
        for j in range(num_n):
            lo = sm*i
            lf = min(sm*(i+1),m)
            ro = sn*j
            rf = min(sn*(j+1),n)
            # h = computeEntropy( im[lo:lf,ro:rf], im[alo:alf, aro:arf] )
            h = computeEntropy( im[lo:lf,ro:rf], I )
            
            imr[lo:lf,ro:rf] = h

    # imw = im.copy()
    # # imw[alo:alf, aro:arf] = im[alo:alf, aro:arf]*10
    # ax[0].imshow( imw,  cmap='gray' )
    # ax[1].imshow( imr )
    return imr


# plotEntropy( im )
# plt.show()
