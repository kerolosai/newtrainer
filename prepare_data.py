import os
import shutil
import pymongo
from azure.storage.blob import BlobClient

from constants import DISEASE_SCIENTIFIC_NAMES, DISEASE_ARABIC_NAMES


valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

# Function to check if a folder has valid images
def has_valid_images(folder_path):
    return any(os.path.splitext(f)[-1].lower() in valid_extensions for f in os.listdir(folder_path))
def sanitize_name(name):
        # Replace invalid characters with underscore 3shan el data feha "other_|"
        return re.sub(r'[<>:"/\\|?*]', '_', name)

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
            edited_crop = doc['editedCrop'].strip()
            edited_crop=sanitize_name(edited_crop)
            edited_disease = doc['editedDisease'].strip()
            edited_disease=sanitize_name(edited_disease)

            #if edited_crop.lower().strip() == 'wheat' or edited_crop.lower().strip() == 'corn':
            print(edited_crop.lower().strip())
            print(edited_disease)
            crop_dir = os.path.join(self.model_data_path, edited_crop)
            # Create a folder for each crop if it doesn't exist
            if not os.path.exists(crop_dir):
                os.makedirs(crop_dir)
            # Create a subfolder for each disease under the crop folder
            disease_common_name = self.get_common_name(edited_disease).strip()
            disease_dir = os.path.join(crop_dir, disease_common_name)
            if not os.path.exists(disease_dir):
                os.makedirs(disease_dir)
            # if not has_valid_images(disease_dir):
            #     print(f"Skipping {disease_dir} because it has no valid images.")
            #     continue
            download_file_path = os.path.join(disease_dir, os.path.basename(image_uri))
            #if the image uri is already exists , it does not download it again
            if os.path.exists(download_file_path):
                print(f"File {os.path.basename(image_uri)} already exists. Skipping download.")
                continue


            
            try:
                self.download_image_from_blob("imagedataset", image_uri.split('/')[-1], download_file_path)
            except:
                pass

    # def create_master_data_directory(self):
    #     if os.path.exists(self.model_data_path):
    #         shutil.rmtree(self.model_data_path)
    #     shutil.copytree(self.base_data_path, self.model_data_path)

    def delete_sparse_folders(self, min_images=1):


        if os.path.isdir(self.model_data_path):
            for crop_name in os.listdir(self.model_data_path):
                crop_path = os.path.join(self.model_data_path, crop_name)
                if os.path.isdir(crop_path):
                    for disease_folder in os.listdir(crop_path):
                        folder_path = os.path.join(crop_path, disease_folder)

                        # Count valid images in the folder
                        valid_image_count = sum(
                            os.path.splitext(f)[-1].lower() in valid_extensions
                            for f in os.listdir(folder_path)
                        )

                        if valid_image_count < min_images:
                            shutil.rmtree(folder_path)
                            print(f"Deleted {folder_path} due to insufficient images.")

    def move_and_rename_folders(self,base_path, disease_dict):
    for crop, diseases in disease_dict.items():
        crop_path = os.path.join(base_path, crop)

        

        #getting the arabic and english folder paths from the inner dictionary 
        for english_name, arabic_name in diseases.items():
            #arabic_name = unicodedata.normalize('NFC', arabic_name).strip()
            arabic_folder_path = os.path.join(crop_path, arabic_name)
            english_folder_path = os.path.join(crop_path, english_name)
            normalized_arabic_name = arabic_name.strip()  # Strip leading/trailing spaces
            normalized_arabic_name = " ".join(normalized_arabic_name.split())  # Replace multiple spaces with a single space
            arabic_folder_path = os.path.join(crop_path, normalized_arabic_name)
            
            # checks if the arabic folder exists , if not do nothing as there is nothing to move
            if os.path.exists(arabic_folder_path):
                # making sure that there is english foloder exists , if not it creates one
                os.makedirs(english_folder_path, exist_ok=True)
                
                # Move files from arabic folder to english  folder
                for file_name in os.listdir(arabic_folder_path):
                    src_file = os.path.join(arabic_folder_path, file_name)
                    dest_file = os.path.join(english_folder_path, file_name)
                    
                    # Moving files steps
                    if os.path.isfile(src_file):
                        shutil.move(src_file, dest_file)
                
                # deleting the arabic folder 
                os.rmdir(arabic_folder_path)
                print(f"Moved contents and deleted folder: {arabic_folder_path}")
            else:
                print(f"No Arabic folder found for disease: {arabic_name}")

    def prepare_data(self):
        # self.create_master_data_directory()
        docs = self.read_all_from_db(
            self.mongo_client,
            {"status": "verified"},
            "diseasedetections",
            sort=[('createdAt', pymongo.DESCENDING)]
        )
        self.download_verified_data(docs)
       # self.delete_sparse_folders()
        self.move_and_rename_folders(self.model_data_path, DISEASE_ARABIC_NAMES)
        return self.model_data_path
