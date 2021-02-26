from math import *
import numpy as np

l = 25.05 # mm
l_curve = 4.5 # mm
h = 25 # mm
a = 50*pi/180 # deg -> rad

d = 0.5 # mm

dist = l - l_curve + ((h/2) + d)/tan(a/2)
dist /= sqrt(2)

vec = [-dist,-dist]
c, s = np.cos(a), np.sin(a)
j = np.matrix([[c, s], [-s, c]])
first = np.dot(j, vec) + [dist,dist]
c, s = np.cos(-a), np.sin(-a)
j = np.matrix([[c, s], [-s, c]])
second = np.dot(j, vec) + [dist,dist]

print('distance = ', dist)
print('r = ', ((h/2) + d)/tan(a/2))
print([dist,dist])
print(first)
print(second)