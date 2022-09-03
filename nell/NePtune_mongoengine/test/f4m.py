from flask import Flask, render_template
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config['DEBUG'] = True  # 开启 debug
mongo = PyMongo(app, uri="mongodb://localhost:27017/traffic")  # 开启数据库实例


@app.route('/user/<string:age>')
def query_user(age):
    if age:
        users = mongo.db.person.find({'age': age})
        print(type(users))
        print(users)
        if users:
            return render_template('user.html', users=users)
        else:
            return 'No user found!'

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)