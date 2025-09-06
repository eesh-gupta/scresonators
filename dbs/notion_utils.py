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


def get_notion_db():
    """
    Returns the Notion database ID from configs/secrets.yml.
    """
    with open("../configs/secrets.yml", "r") as file:
        secrets = yaml.safe_load(file)
    name_db = secrets["notion_db_id"]
    n2p = get_notion_client()
    notion_df = n2p.from_notion_DB_to_dataframe(name_db)
    return notion_df


def send_notion_db(notion_df):
    """
    Sends the updated Notion dataframe back to the Notion database.
    """
    n2p = get_notion_client()
    with open("../configs/secrets.yml", "r") as file:
        secrets = yaml.safe_load(file)
    name_db = secrets["notion_db_id"]
    result = n2p.update_notion_DB_from_dataframe(name_db, notion_df)
    return result
