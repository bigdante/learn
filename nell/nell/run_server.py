from flask import Flask, abort, request, jsonify
from datetime import datetime
from settings import *
from flask_cors import cross_origin, CORS
import subprocess
from data_object import *
import random
# from example.read_from_db import read_examples_from_db, thumb_up_to_db, thumb_down_to_db, get_records_num, \
#     get_running_days
import datetime

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)


# def CMD(command, wait=True):
#     h = subprocess.Popen(command, shell=True)

@app.route('/latest', methods=['GET'])
def latest():
    result = []
    for index, triple in enumerate(TripleFact.objects()):
        result.append(precess_db_data(triple))
        # show only 50
        if index >=500:
            break
    random.shuffle(result)
    # return json.dumps(result[:50], indent=2)
    return output_process(result[:50])


@app.route('/dashboard', methods=['GET'])
def dashboard():
    # ent_num, rel_num = tuple(open('get_ent_rel_num.txt').readlines())
    # ent_num = int(ent_num.strip('\n'))
    # rel_num = int(rel_num.strip('\n'))
    return output_process(
        {"all_triples": random.randint(0,10), "all_entities": random.randint(0,10), "all_relations": random.randint(0,10),
         'running_days': random.randint(0,10)})

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return str(o)
        return json.JSONEncoder.default(self, o)


e = JSONEncoder()

def output_process(result):
    result = json.loads(e.encode(result))
    return jsonify(result)

def precess_db_data(db_document):
    output = {}
    output['_id']= db_document.id
    output['head_entity']= db_document.head
    output["head_linked_entity"]="????"
    output['relation']=db_document.relationLabel
    output['tail_entity']=db_document.tail
    output['evidences']=[{
            "text" : db_document.evidenceText,
            "extractor" : "GLM-2B/P-tuning",
            "confidence" : random.random(),
            "filtered" : True,
            "ts" : str(datetime.date.today())
    }]

    return output


if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    # CMD("python3 run.py")
    app.run(host="0.0.0.0", port=8841, debug=True)
