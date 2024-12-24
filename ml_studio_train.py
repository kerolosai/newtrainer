import warnings

warnings.filterwarnings('ignore')
import os
import shutil
import torch
from datetime import datetime
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile

from src.helper import split_train_test, train_transform, test_transform
from src.constants import LEARNING_RATE, NUM_EPOCHS
from src.azure_utils import upload_data_to_blob
from src.prepare_data import DataPreparation

ImageFile.LOAD_TRUNCATED_IMAGES = True
checkpoint_folder = f"C://Users/mjahaseel/Desktop/trainer_checkpoint"
mongo_client = MongoClient(os.getenv('MONGODB_CONNECTION_STRING'))


class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except (IOError, SyntaxError) as e:
            # print(f"Skipping corrupted image: {path}")
            return self.__getitem__((index + 1) % len(self.samples))


class Trainer:
    def __init__(self, model_id, mongo_client):
        self.model_id = model_id
        self.mongo_client = mongo_client

    def calculate_correct(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        return correct
    
    def prepare_data(self):
        data_preparation_obj = DataPreparation(self.model_id, self.mongo_client)
        data_dir = data_preparation_obj.prepare_data()
        return data_dir
    
    def save_to_db(self, model_info_for_db):
        db = self.mongo_client["hudhud"]
        collection = db["diseasedetectionmodels"]
        result = collection.insert_one(model_info_for_db)
        _id = str(result.inserted_id)
        return _id

    def train(self, final_train=False, final_train_data_dir=None, train_data_dir=None, metrics=None):
        if not final_train:
            # prepare data
            print("\nPreparing data ...")
            data_dir = self.prepare_data()
            print("\nData Directory: ", data_dir)
            print("Data preparation completed successfully !")



            #iterating over each crop folder:
            for crop_name in os.listdir(data_dir):

                crop_folder_path = os.path.join(data_dir, crop_name)
                if os.path.isdir(crop_folder_path):
                    print(f"\nPreparing data for crop: {crop_name} ...")
                    # checkpoint_folder_for_each_crop = f"C://Users/mjahaseel/Desktop/trainer_checkpoint/{crop_name}"
                    # if not os.path.exists(checkpoint_folder_for_each_crop):
                    #     os.makedirs(checkpoint_folder_for_each_crop)
                    # model_path =  checkpoint_folder_for_each_crop+f'/{crop_name}.pth'
                    # Split the training and validation data
                    TRAINING_DATA_PATH = f'src/data/training_data'/{crop_name}'
                    train_data_dir = split_train_test(crop_folder_path, TRAINING_DATA_PATH)
                    print(f"Training data for {crop_name} split completed!")
                    train_images = SafeImageFolder(f'{train_data_dir}/train', transform=train_transform())
                    val_images = SafeImageFolder(f'{train_data_dir}/val', transform=test_transform())
                    #train/val data loader:
                    trainloader = torch.utils.data.DataLoader(train_images, batch_size=32, shuffle=True, num_workers=0)
                    valloader = torch.utils.data.DataLoader(val_images, batch_size=32, num_workers=0)
                    #getting the classs and its number
                    classes = train_images.classes
                    num_classes = len(classes)



                    AlexNet_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
                    AlexNet_model.classifier[6] = nn.Linear(4096, num_classes)
                    AlexNet_model.eval()

                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    # device = torch.device("cpu")
                    AlexNet_model.to(device)

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(AlexNet_model.parameters(), lr=LEARNING_RATE, momentum=0.9)

                    print("\nStarting training ...\n")
                    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
                        running_loss = 0
                        running_corrects = 0
                        total_train = 0
                        for i, data in enumerate(trainloader, 0):
                            # get the inputs; data is a list of [inputs, labels]
                            inputs, labels = data[0].to(device), data[1].to(device)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward + backward + optimize
                            output = AlexNet_model(inputs)
                            loss = criterion(output, labels)
                            loss.backward()
                            optimizer.step()

                            # Calculate accuracy
                            correct = self.calculate_correct(output, labels)

                            # print statistics
                            running_loss += loss.item()
                            running_corrects += correct
                            total_train += labels.size(0)

                            if i % 40 == 0:
                                print('[%d, %5d] loss: %.3f' % (epoch + 1, i, running_loss / 40))
                                running_loss = 0
                        train_accuracy = (100 * running_corrects / total_train)
                        running_loss /= len(trainloader)

                        # Validation loop
                        val_loss = 0
                        val_corrects = 0
                        total_val = 0

                        AlexNet_model.eval()  # Set the model to evaluation mode
                        with torch.no_grad():
                            for val_data in valloader:
                                val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                                val_outputs = AlexNet_model(val_inputs)
                                val_loss += criterion(val_outputs, val_labels).item()

                                # Calculate accuracy
                                val_correct = self.calculate_correct(val_outputs, val_labels)
                                val_corrects += val_correct
                                total_val += val_labels.size(0)

                        val_loss /= len(valloader)
                        val_accuracy = (100 * val_corrects / total_val)

                        print(f'Epoch {epoch + 1} Training Loss: {running_loss:.3f}, Training Accuracy: {train_accuracy:.2f}%, '
                            f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%')

                        # Set the model back to training mode
                        AlexNet_model.train()

                        # if epoch % 2 == 0:
                        #     checkpoint = {
                        #         'state_dict': AlexNet_model.state_dict(),
                        #         'class_to_idx': train_images.class_to_idx,
                        #         'num_classes': num_classes,
                        #         'device': device
                        #     }
                        #     torch.save(checkpoint, f'all_crops_model_29052024_epoch_{epoch + 1}.pth')

                    checkpoint = {
                        'state_dict': AlexNet_model.state_dict(),
                        'class_to_idx': train_images.class_to_idx,
                        'num_classes': num_classes,
                        'device': device
                    }

                    #torch.save(checkpoint, model_path)

                    if final_train:
                        metrics = {
                            "trainingAccuracy": train_accuracy,
                            "testAccuracy": val_accuracy
                        }
                        # self.train(final_train=True, final_train_data_dir=data_dir, train_data_dir=train_data_dir, metrics=metrics)
                        final_train = False

                    if not final_train:
                        model_url = upload_data_to_blob("disease-detection-models", self.model_id, checkpoint)
                        model_info = {
                            "modelId": f"{self.model_id}_{crop_name}",
                            #"modelUrl": model_url,
                            "status": "inactive",
                            "createdAt": datetime.utcnow(),
                            "modifiedAt": datetime.utcnow(),
                            "metrics": metrics
                        }
                        doc_id = self.save_to_db(model_info_for_db=model_info)
                        print("\nTraining completed successfully ...\n")


if __name__ == '__main__':
    modelNameSuffix = str(datetime.today().date())
    model_id = model_id + '-' + modelNameSuffix
    trainer_obj = Trainer(model_id, mongo_client)
    trainer_obj.train()