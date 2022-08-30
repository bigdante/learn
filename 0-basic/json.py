import json
with open('../8-ner_nre/pytorch-master/train_data_me.json', 'r') as file:
    str = file.read()
    data = json.loads(str)
    for i in data:
        print(i)
    # print(data)