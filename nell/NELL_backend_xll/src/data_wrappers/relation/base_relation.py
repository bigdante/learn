from typing import List, Dict


class BaseRelation:
    def __init__(self, _id: str, surface: str, **kwargs):
        self.id: str = _id
        self.surface: str = surface
        self.meta: Dict = kwargs

    def get_cls_name(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return "[{}: {}]".format(self.get_cls_name(), self.surface)
