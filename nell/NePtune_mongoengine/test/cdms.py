# 对本地的mongodb进行增删改查操作

import pymongo
client = pymongo.MongoClient(host='127.0.0.1')
# 如果数据库没有，就相当于准备创建一个
db = client.traffic  # 数据库名为 traffic
print(db)
db_list = client.list_database_names()
print(db_list)
collections = db.person
person_one = {
    'name': '咪哥杂谈',
    'age': '24',
    'height': '175',
    'weight': '60'
}
result = collections.insert_one(person_one)  # 文档插入集合
print(result)  # 打印结果
print(result.inserted_id)  # 打印插入数据的返回 id 标识
db_list = client.list_database_names()
print(db_list)

person_one = {
    'name': '咪哥杂谈',
    'age': '24',
    'height': '175',
    'weight': '60'
}

person_two = {
    'name': '咪哥杂谈_two',
    'age': '22',
    'height': '180',
    'weight': '63'
}

result = collections.insert_many([person_one, person_two])  # 文档插入集合
print(result)
print(result.inserted_ids)
# 查
result = collections.find()

for r in result:
    print(r)
print("=========================")
# 0代表不输出
result = collections.find({}, {'_id': 0, 'name': 1, 'age': 1})
for a in result:
    print(a)
print("*************************")
result = collections.find({}, {'name': 1, 'age': 1})
for a in result:
    print(a)

result2 = collections.find({'age': '22'})
for a in result2:
    print(a)

# 改
print('更新前..........')
for v in collections.find():
    print(v)

query_name = {"name": "咪哥杂谈"}
new_value = {"$set": {"age": "100"}}

collections.update_one(query_name, new_value)
# query_name = {"name": "咪哥杂谈"}
# new_value = {"$set": {"age": "100000"}}
# collections.update_many(query_name, new_value) # 更新多条
print('更新后..........')
for v in collections.find():
    print(v)

# 删除
query_name = {"name": "咪哥杂谈_three"}
collections.delete_one(query_name)  # 删除单条
query_name = {"name": "咪哥杂谈"}
collections.delete_many(query_name) # 删除多条
collections.delete_many({})  # 删除所有数据
