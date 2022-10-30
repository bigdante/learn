import random
from mongoengine import *

connect(host="mongodb://localhost:27017/", connect=False)  # 需要有默认连接
connect(host="mongodb://localhost:27017/userInfo", alias="userInfo", connect=False)

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

if __name__ == '__main__':

    user = UserInfo.objects.get(_id=1)
    print(user._created)
    user.age2 = 2  # 修改年龄
    user.nick_name = "test"  # 增加昵称字段
    a = user.save()
    print(a.age)
    print(a._created)

