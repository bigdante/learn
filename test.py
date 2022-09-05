i = {(20, 34): [(1, 2, 6)], (21, 35): [(1, 2, 6)]}
from random import choice

a = list(i.keys())
print(a)
k1, k2 = choice(list(i.keys()))
print(k1, k2)
