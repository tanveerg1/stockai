from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os

# Get MongoDB credentials from environment variables
db_username = os.getenv("MONGO_DB_USERNAME")
db_password = os.getenv("MONGO_DB_PASSWORD")
cluster_name = os.getenv("MONGO_DB_CLUSTER_NAME")
app_name = os.getenv("MONGO_DB_APP_NAME")   

uri = f"mongodb+srv://{db_username}:{db_password}@stockaicluster.{cluster_name}.mongodb.net/?retryWrites=true&w=majority&appName={app_name}"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
