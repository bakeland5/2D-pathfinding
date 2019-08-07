import numpy as np
import numpy.linalg as la


def dobj(P,r,c1,c2):
    eps = 10**(-6)
    n,_ = P.shape
    k,_ = r.shape
    G = np.zeros((n,2))
    for j in range(1,n-1):
        x = P[j,:]
        xim1 = P[j-1,:]
        xip1 = P[j+1,:]
        t1 = 0
        for objt in range(k):
          t1 += -2*c1*(x-r[objt])/(eps+(la.norm(x-r[objt]))**2)**2
        t2 = 2*c2*(2*x-xim1-xip1)
        G[j, :] = t1+t2
    return G


def obj(P, r, c1, c2):
    eps = 10**(-6)
    n,_ = P.shape
    k,_ = r.shape
    t1 = 0
    t2 = 0
    for i in range(n):
        x = P[i,:]
        for j in range(1,k-1):
            t1 += c1/(eps+la.norm(x-r[j])**2)
    for i in range(n-1):
        x = P[i,:]
        xip1 = P[i+1,:]
        t2 += c2*la.norm(xip1-x)**2
    f = t1 + t2
    return f
