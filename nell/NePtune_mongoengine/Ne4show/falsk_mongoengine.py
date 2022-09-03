from flask import Flask
from flask_mongoengine import MongoEngine
import configparser
from data_object.relation.base_relation import BaseRelation

db = MongoEngine()


def create_app():
    app = Flask(__name__)
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    mongodb = dict(config.items('mongodb'))

    app.config['MONGODB_SETTINGS'] = {
        'db': mongodb.get('db'),
        'username': mongodb.get('username'),
        'password': mongodb.get('password'),
        'host': mongodb.get('host'),
        'port': int(mongodb.get('port')),
        'authentication_source': mongodb.get('authentication_source')
    }
    app.config['SECRET_KEY'] = 'dante'
    db.init_app(app)
    return app

app = create_app()


@app.route('/')
@app.route('/nell4show')
def nell4show():
    # 前端界面，用户点击的过程中，后台接收到变量，处理后将数据库内容送回界面并展示，这里应该是要用到异步
    # for i in BaseRelation.objects:
    #     print(i)
    return "Hello, World!"

# 用flask对mongodb进行增删改查，前端输入一个变量，这里要进行接收和处理

if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="0.0.0.0", port=8841, debug=True)
