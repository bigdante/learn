import os

path = "./dd/aa"
a = 1
bp = os.path.join(path, str(a))
if not os.path.exists(path):
    os.makedirs(bp)

f = open(os.path.join(bp, f"{a}.txt"), "w")
