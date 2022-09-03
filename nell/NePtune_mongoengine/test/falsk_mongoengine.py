from flask_mongoengine import MongoEngine
from flask import Flask
from client.mongo_ssh import MongoDB

MongoDB().connect()
import json
from person import Person
from userinfor import UserInfo
from data_object import *


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dante'
    return app


app = create_app()


@app.route('/user/<int:age>')
def query_user(age):
    result = []
    for a in UserInfo.objects(age=age):
        # print(a.country)
        result.append([getattr(a, i) for i in a.__dict__['_fields_ordered']])
    for a in Person.objects(height="175"):
        print(a.name)
    for a in BaseRelation.objects(text='place of birth'):
        print(a)
        print(a.HeadConstraint)
        # return a.description
    # idms

    return json.dumps(result, indent=2)


if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="0.0.0.0", port=8841, debug=True)
