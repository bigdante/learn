from flask import Flask, abort, request, jsonify
from datetime import datetime
from settings import *
from flask_cors import cross_origin, CORS
import subprocess

# from example.read_from_db import read_examples_from_db, thumb_up_to_db, thumb_down_to_db, get_records_num, \
#     get_running_days

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)


def CMD(command, wait=True):
    h = subprocess.Popen(command, shell=True)


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


@app.route('/thumb_up', methods=['POST'])
def thumb_up():
    data = request.get_json()
    _id = data.get('_id')
    action = data.get('action')
    thumb_up_to_db(_id, action)
    return output_process({"status": "success"})


@app.route('/thumb_down', methods=['GET'])
def thumb_down():
    data = request.get_json()
    _id = data.get('_id')
    action = data.get('action')
    thumb_down_to_db(_id, action)
    return output_process({"status": "success"})


@app.route('/latest', methods=['GET'])
def latest():
    return output_process(read_examples_from_db())


@app.route('/dashboard', methods=['GET'])
def dashboard():
    ent_num, rel_num = tuple(open('get_ent_rel_num.txt').readlines())
    ent_num = int(ent_num.strip('\n'))
    rel_num = int(rel_num.strip('\n'))
    return output_process(
        {"all_triples": get_records_num(), "all_entities": ent_num, "all_relations": rel_num,
         'running_days': get_running_days()})


if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    CMD("python3 run.py")
    app.run(host="0.0.0.0", port=8841, debug=True)
