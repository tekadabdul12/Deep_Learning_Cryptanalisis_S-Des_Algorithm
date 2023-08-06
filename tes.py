import numpy as np

print(bin(3))

#print (bin(0b000011 * 0b00010))

a = np.array([3.],dtype=float)

#for i in a:
#    print(np.binary_repr(i, width=4))

b = a.astype(np.float32)
print(b, b.dtype)
b = b.astype(int)
print(b , b.dtype, b.shape)