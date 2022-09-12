from flask import Flask, abort, request, jsonify
from datetime import datetime
from bson import ObjectId
from mongoengine.queryset.visitor import Q

from settings import *
from flask_cors import cross_origin, CORS
from data_object import *
import random

import datetime

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)


@app.route('/latest', methods=['GET'])
def latest():
    result = []
    for index, triple in enumerate(TripleFact.objects()):
        result.append(precess_db_data(triple))
        # show only 50
        if index >= 500:
            break
    random.shuffle(result)
    return output_process(result[:50])


@app.route('/dashboard', methods=['GET'])
def dashboard():
    # ent_num, rel_num = tuple(open('get_ent_rel_num.txt').readlines())
    # ent_num = int(ent_num.strip('\n'))
    # rel_num = int(rel_num.strip('\n'))
    return output_process(
        {"all_triples": random.randint(0, 10), "all_entities": random.randint(0, 10),
         "all_relations": random.randint(0, 10),
         'running_days': random.randint(0, 10)})


@app.route('/pps', methods=['GET', 'POST'])
def show_pps():
    # query
    # if request.method == 'POST':
    # 	argsJson = request.data.decode('utf-8')
    # 	argsJson = json.loads(argsJson)
    # 	print(argsJson)
    # 	result = argsJson
    # 	result = json.dumps(result, ensure_ascii=False)	#转化为字符串格式
    # 	return result									#return会直接把处理好的数据返回给前端
    # else:
    # 	return " 'it's not a POST operation! "
    #   获取到某个page的ID，在此之前应该通过某个字段谋取到对应的page，并获取到page的id
    # 这里先设置page id为 624989b1c20df149acb246cf
    result_triples = {}
    result_page_ids = {}
    for index, page in enumerate(WikipediaPage.objects()):
        # get sentences ids of this page
        result_ids = []
        for  paragrah in page.paragraphs:
            for sentence in paragrah.sentences:
                result_ids.append(sentence.id)
        result_page_ids[index]=result_ids
        print(result_ids)
    print("get all pages")
    for index, page_ids in result_page_ids.items():
        result_list = []
        for id in page_ids:
            result = []
            for triple in TripleFact.objects(evidence=id):
                # print(triple.evidence.text)
                result.append(precess_db_data(triple))
            if result:
                result_list.append(result)
        if result_list:
            result_triples[index]=result_list
            print(result_list)
            # break
        else:
            print("holly shit, no result in this page")

        if result_triples and len(result_triples)>50:
            print(result_triples)
            with open('result_triples.json', 'w') as f:
                json.dump(result_triples,f)
            break
    return output_process(result_triples)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return str(o)
        return json.JSONEncoder.default(self, o)


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
    output['_id'] = db_document.id
    output['head_entity'] = db_document.head
    output["head_linked_entity"] = "????"
    output['relation'] = db_document.relationLabel
    output['tail_entity'] = db_document.tail
    output['evidences'] = [{
        "text": db_document.evidenceText,
        "extractor": "GLM-2B/P-tuning",
        "confidence": random.random(),
        "filtered": True,
        "ts": str(datetime.date.today())
    }]

    return output


if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    # CMD("python3 run.py")
    app.run(host="0.0.0.0", port=8841, debug=True)
