import os
import shutil
import pymongo
from azure.storage.blob import BlobClient

from constants import DISEASE_SCIENTIFIC_NAMES, DISEASE_ARABIC_NAMES


class DataPreparation:
    def __init__(self, model_id, mongo_client):
        self.model_id = model_id
        self.mongo_client = mongo_client
        # self.base_data_path = 'src/data/base_data'
        # self.model_data_path = os.path.join('src/data', self.model_id)
        self.model_data_path = 'src/data/base_data'

    def download_image_from_blob(self, container_name, blob_file_path, save_local_path=None):
        connection_string = f'DefaultEndpointsProtocol=https;AccountName={os.getenv("storage_account_name")};AccountKey={os.getenv("storage_account_key")}'
        blob_client = BlobClient.from_connection_string(conn_str=connection_string, container_name=container_name, blob_name=blob_file_path)
        with open(save_local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

    def read_all_from_db(self, mongo_client, query, collection_name, sort=None, limit=None):
        db = mongo_client["hudhud"]
        collection = db[collection_name]
        if not sort and not limit:
            cursor = collection.find(query)
        elif sort and limit:
            cursor = collection.find(query).sort(sort).limit(limit)
        elif sort:
            cursor = collection.find(query).sort(sort)
        elif limit:
            cursor = collection.find(query).limit(limit)
        results = [data for data in cursor]
        return results
    
    def get_common_name(self, disease_name):
        # Check in scientific names
        for common_name, sci_name in DISEASE_SCIENTIFIC_NAMES['wheat'].items():
            if disease_name.lower() in sci_name.lower():
                return common_name
        # Check in Arabic names
        for common_name, arabic_name in DISEASE_ARABIC_NAMES['wheat'].items():
            if disease_name.lower() in arabic_name.lower():
                return common_name
        return disease_name

    def download_verified_data(self, docs):
        existing_directories = {d.lower(): d for d in os.listdir(self.model_data_path)}

        for doc in docs:
            image_uri = doc['imageUri']
            edited_crop = doc['editedCrop']
            edited_disease = doc['editedDisease'].lower().strip()

            if edited_crop.lower().strip() == 'wheat' or edited_crop.lower().strip() == 'corn':
                disease_common_name = self.get_common_name(edited_disease).lower()
                class_dir_path = f"{edited_crop.lower().strip()}_{disease_common_name}"
                class_dir = os.path.join(self.model_data_path, existing_directories.get(class_dir_path, class_dir_path))

                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                    existing_directories[class_dir_path] = os.path.basename(class_dir)

                download_file_path = os.path.join(class_dir, os.path.basename(image_uri))
                try:
                    self.download_image_from_blob("imagedataset", image_uri.split('/')[-1], download_file_path)
                except:
                    pass

    # def create_master_data_directory(self):
    #     if os.path.exists(self.model_data_path):
    #         shutil.rmtree(self.model_data_path)
    #     shutil.copytree(self.base_data_path, self.model_data_path)

    def delete_sparse_folders(self):
        # for crop_name in os.listdir(self.model_data_path):
            # crop_path = os.path.join(self.model_data_path, crop_name)
        if os.path.isdir(self.model_data_path):
            for disease_folder in os.listdir(self.model_data_path):
                folder_path = os.path.join(self.model_data_path, disease_folder)
                if os.path.isdir(folder_path) and len(os.listdir(folder_path)) < 50:
                    shutil.rmtree(folder_path)
                    print(f"Deleted {folder_path} due to insufficient images")

    def prepare_data(self):
        # self.create_master_data_directory()
        docs = self.read_all_from_db(
            self.mongo_client,
            {"status": "verified"},
            "diseasedetections",
            sort=[('createdAt', pymongo.DESCENDING)]
        )
        self.download_verified_data(docs)
        self.delete_sparse_folders()
        return self.model_data_path
