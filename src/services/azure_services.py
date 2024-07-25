from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, BlobSasPermissions

from datetime import datetime, timedelta

from utils.function import generate_random_hex

import traceback
import sys
import os 


class azure_ops:
    """
    Represents operations related to Azure Blob Storage.
    """
    def __init__(self, account_name, account_key, container_name, blob_path):
        
        """
        Initializes an AzureOps instance.

        Args:
            account_name (str): Azure Storage account name.
            account_key (str): Azure Storage account key.
            container_name (str): Name of the container.
            blob_path (str): Path to the blob.
        """

        try:
            self.account_name = account_name
            self.account_key = account_key
            self.container_name = container_name
            self.blob_path = blob_path
            
            self.connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + self.account_name + ';AccountKey=' + self.account_key + ';Endpointsuffix=core.windows.net'
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
            print("azure connected sucessfully !!")
        except Exception as e:
            print(f'Exception occured on connecting to Azure portal inside azure_services: {e}')
            print(traceback.print_exc(file=sys.stdout))
            
    def generate_sas_url(self, file_name: str):
        """
        Generates a shared access signature (SAS) URL for the specified blob.

        Args:
            file_name (str): Name of the blob.

        Returns:
            str: SAS URL for the blob.
        """
        try:
            sas_url = ''
            for blob_i in self.container_client.list_blobs(name_starts_with=self.blob_path):
                if blob_i.name.split('/')[-1]== file_name:
                    sas_i = generate_blob_sas(account_name = self.account_name,
                                    container_name = self.container_name,
                                    blob_name = blob_i.name,
                                    account_key=self.account_key,
                                    permission=BlobSasPermissions(read=True),
                                    expiry=datetime.utcnow() + timedelta(hours=1))
                    sas_url = 'https://' + self.account_name+'.blob.core.windows.net/' + self.container_name + '/' + blob_i.name + '?' + sas_i
            return sas_url
        except Exception as e:
            print(f'Exception occured while generating sas token in Azure portal inside azure_services: {e}')
            print(traceback.print_exc(file=sys.stdout))
    
    def check_blob_and_rename(self, file_name):        
        """
    Checks if the blob exists and renames the new file if needed.

    Args:
        file_name (str): Name of the blob.

    Returns:
        str: New file name (with a counter) if the blob exists, otherwise the original file name.
    """
        # Check if the blob exists
        blob_list = self.container_client.list_blobs(name_starts_with=self.blob_path)
        blob_exists = False
        for blob in blob_list:
            if blob.name.split('/')[-1] == file_name:
                blob_exists = True
                break
        
        # If the blob exists, rename the new file
        if blob_exists:
            # Generate a new file name by appending a number to the original file name
            base_name, file_extension = os.path.splitext(file_name)
            counter = 1
            new_file_name = f"{base_name}_{counter}{file_extension}"
            
            # Check if the new file name already exists
            while any(blob.name.split('/')[-1] == new_file_name for blob in blob_list):
                counter += 1
                new_file_name = f"{base_name}_{counter}{file_extension}"
            # Return the new file name
            return new_file_name
        else:
            # If the blob does not exist, return the original file name
            return file_name
        
    def upload_file(self, local_file_path, file_name):
        try:
            rand_num = generate_random_hex(length=16)
            file_name = f'{file_name.split(".")[0]}_{rand_num}.{file_name.split(".")[-1]}'
            az_file_name = self.check_blob_and_rename(file_name)
            with open(local_file_path, 'rb') as data:
                blob_client = self.container_client.upload_blob(name=self.blob_path+az_file_name, data=data)
            sas_url = self.generate_sas_url(file_name=az_file_name)
            return sas_url
        except Exception as e:
            print(f'Exception occured while uploading file to Azure portal: {e}')
            print(traceback.print_exc(file=sys.stdout))
    
    def download_blob(self, file_name, download_file_path):
        try:
            # Create a blob client
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=self.blob_path+file_name)
            print('azure file downloaded')
            # Download the blob
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        except Exception as e:
            print(f'Exception occured while downloading file from Azure portal: {e}')
            print(traceback.print_exc(file=sys.stdout))
            
            
    def azure_close_conn(self):
        try:
            self.blob_service_client.close()
            print("Connection Closed")
        except Exception as e:
            print(f'Exception occured while closing azure services: {e}')
            print(traceback.print_exc(file=sys.stdout))
        