import random
from mongoengine import *
# connect(host="mongodb://localhost:27017/", connect=False)  # 需要有默认连接
# connect(host="mongodb://localhost:27017/userInfo", alias="userInfo", connect=False)
class UserInfo(DynamicDocument):
    """
    用户游戏任务相关信息
    """
    uid = SequenceField(primary_key=True)  # 自增id
    create_time = StringField(max_length=100)  # 账号创建时间
    device_id = StringField(max_length=100)  # 设备id
    city = StringField(max_length=50)  # 城市
    gender = IntField()  # 性别
    province = StringField(max_length=50)  # 省份
    country = StringField(max_length=50)  # 国家
    headimg = StringField(max_length=200)  # 头像
    role = StringField(max_length=100)  # 用户身份
    nick_name = StringField(max_length=100)  # 昵称
    phone = StringField(max_length=100)  # 手机号
    real_name = StringField(max_length=100)  # 真名
    id_card = StringField(max_length=200)  # 身份证号
    # shard_key 为分片key，在使用mongo分片集群的时候需要配置并且在项目首次上线之前在mongo中手动声明分片库和分片集合
    meta = {"collection": "userInfo", "db_alias": "userInfo", "shard_key": ("uid",)}



# import time
# user = UserInfo(create_time=time.strftime("%Y-%m-%d %H:%M:%S"), device_id="21c267a9-c04d-38de-a0c0-641366ed5e43")
# user.country = "中国"
# user.city = "北京"
# user.age = 18
# user.save()
# # 插入后的示例数据 主键uid在mongo中为_id
# # { "_id" : 1, "create_time" : "2020-03-04 09:06:28", "device_id" : "21c267a9-c04d-38de-a0c0-641366ed5e43", "city" : "北京", "country" : "中国",age: 18 }
#
# # 查单条数据，如果没有会报错
# user = UserInfo.objects.get(_id=1)
# print(user.uid, user.country, user.city)
# # 查询多条，返回列表,如果没有数据返回空列表
# user = UserInfo.objects(_id=1)
# for i in user:
#     print(i.uid, i.country, i.city)
# # 操作符格式 字段__操作符=查询条件
# # 创建时间大于2020-03-04 12:07:04的数据
# user = UserInfo.objects(create_time__gte="2020-03-04 12:07:04")
# for i in user:
#     print(i.uid, i.country, i.city)
# '''
# 常用操作符
# * ne 不等于  age__ne=18
# * gt(e) 大于(等于) create_time__gte="2020-03-04 12:07:04"
# * lt(e) 小于(等于) create_time__lte="2020-03-04 12:07:04"
# * not 对操作符取反，比如 age__not__gt=18
# * in 后面是一个列表，比如 city__in=["北京"，"上海"],找出这两个城市的数据，若都不存在，返回空列表。
# * nin in取反  age__nin=[18]
# * mod 取模，比如 age__mod=(2,0) 表示查询出age除以2，余数是0的数据
# '''
# # 通过Q对象查询多个条件 |或，& 与
# from mongoengine.queryset.visitor import Q
# user = UserInfo.objects(Q(city="北京")|Q(age=17))
# for i in user:
#     print(i.uid, i.country, i.city, getattr(i, "age", 0))
#
# user = UserInfo.objects.get(_id=1)
# user.age = 20  # 修改年龄
# user.nick_name = "test"  #增加昵称字段
# user.save()
#
# user = UserInfo.objects(_id="10270521").delete()

