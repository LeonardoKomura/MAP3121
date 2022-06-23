import matplotlib.pyplot as plt
import numpy as np
import math

with open('input-a.txt', 'r') as f:
    l = [[float(num) for num in line.split('. ')] for line in f]

pos1 = l[0]
n = int(pos1[0])
A = np.eye(n)
print(n)

vetor1 = l[1]
vetor2 = l[2]
vetor3 = l[3]
vetor4 = l[4]

for i in range(n):
    A[0][i] = vetor1[i]
for i in range(n):
    A[1][i] = vetor2[i]
for i in range(n):
    A[2][i] = vetor3[i]
for i in range(n):
    A[3][i] = vetor4[i]


#input = np.loadtxt("input-a.txt", dtype='f', delimiter='.')
#print(input)


#a = [1, 2, 3]
#b = [1, 2, 1]
#f = [[1, 2, 3],[2, 4, 5],[3,5,6]]
#c = np.transpose(np.array([a]))
#d = np.transpose(np.array([b]))
#e = np.multiply(a, f)
#print(e)
#g = np.multiply(e, a)
#print(g)

#maxvalue=max(a)
#minvalue=min(a)
#
#if abs(maxvalue) > abs(minvalue):
#    maximo = maxvalue
#else:
#    maximo = minvalue
#
#print("valor maximo (em modulo): ", maximo)
#pos = a.index(maximo)
#print("posicao: ", pos)


#c = np.add(a, b)
#d = np.zeros(3)
#for i in range(3):
#    d[i] = c[i]
#print(c)
#print(d)
#
#e = np.subtract(c, d)
#print(e)
#
#
#
#for i in range(4, 1, -1):
#    print(i)