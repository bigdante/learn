from .base_page import *


class WikipediaPage(BasePage):
    """
    Wikipedia pages
    """
    # source
    source = StringField(default="wikipedia")
    source_id = StringField(required=True)
    meta = {
        "collection": "relation",
        # "indexes": [
        #     "sourceId",
        #     "$text"
        # ],
        "db_alias": "NePtune"
    }
