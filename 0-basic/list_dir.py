import os
path = "."
data_name = list(map(lambda x: path + '/' + x, os.listdir(path)))
print(data_name)