from azure.storage.blob import BlobServiceClient
from azure_config import AZURE_STORAGE_CONNECTION_STRING

def upload_to_blob(file_data, filename, container_name="uploads"):
    try:
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service.get_blob_client(container=container_name, blob=filename)
        blob_client.upload_blob(file_data, overwrite=True)
        return blob_client.url
    except Exception as e:
        return f"❌ Error uploading to Azure Blob: {e}"
