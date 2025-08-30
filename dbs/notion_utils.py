import os
import yaml
from notion2pandas import Notion2PandasClient


def get_notion_client():
    """
    Initializes and returns a Notion2PandasClient using the token from configs/secrets.yml.
    """
    with open("../configs/secrets.yml", "r") as file:
        secrets = yaml.safe_load(file)
    token = secrets["notion_token"]
    return Notion2PandasClient(auth=token)
