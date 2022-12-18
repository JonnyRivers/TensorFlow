import numpy as np

a = np.arange(6)
a2 = a[np.newaxis, :]
print(a2.shape)

b = np.array([ [1,2,3,4], [5,6,7,8], [9,10,11,12] ])
print(b.shape)
print(b[1][3]) # -> 8

linspace = np.linspace(0, 10, num=5)
# array([ 0. , 2.5, 5., 7.5, 10. ])

first_two = linspace[:2]
last_two = linspace[-2:]
print(last_two)

c = np.arange(10)
c2 = c.reshape(5,2)
print(c2)