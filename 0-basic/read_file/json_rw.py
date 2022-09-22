# 读
import json

with open('../8-ner_nre/pytorch_nn_rnn-master/train_data_me.json', 'r') as file:
    str = file.read()
    data = json.loads(str)
    for i in data:
        print(i)

# 写
result_triples = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}

with open('result_triples.json', 'w') as f:
    json.dump(result_triples, f)
