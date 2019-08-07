import scipy.optimize as sopt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pt
from grad_objs import obj
from grad_objs import dobj

n = 10
k = 20

init_path = np.array([(1.0, 1.0), (1.3, 4.5), (2.4, 3.5), (4.3, 5.1), (5.2, 5.5), (5.8, 4.9), (6.8, 6.0), (7.9, 6.5), (8.2, 7.5), (8.7, 9.3)], dtype= np.float64)
obstacles = 10*np.random.rand(k,2)
r = obstacles
x = init_path
c2 = 1

print_list = [9,24,49,99,199,399]

pt.figure(figsize=(8,8))
pt.plot(x[:,0], x[:,1], label="Initial")


for iter in range(400):
    c1 = 1000/(100+iter)
    def f(a):
        arg = x-a*dobj(x,r,c1,c2)
        return(obj(arg,r,c1,c2))
    a = sopt.golden(f)
    x = x - a*dobj(x,r,c1,c2)
    if iter in print_list:
        label = 'iter = %s' % (iter+1)
        pt.plot(x[:,0], x[:,1], label=label)


pt.plot(obstacles[:,0], obstacles[:, 1], "o", markersize=5, label="Obstacles")
pt.legend(loc="best")
pt.show()
