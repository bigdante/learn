a = []
a.extend([{"a": 1}, {"b": 1}])
b = {"a": 2}
if b in a:
    print("hhh")
else:
    a.extend([b])
print(a)