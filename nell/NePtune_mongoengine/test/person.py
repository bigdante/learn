from mongoengine import *

class Person(DynamicDocument):
    uid = SequenceField(primary_key=True)  # 自增id
    name = StringField(required=True)
    age = StringField(required=True)
    height = StringField()
    weight = StringField()

    meta = {
        "db_alias": "traffic",
        "collection": "person",
    }

