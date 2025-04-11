"""
https://medium.com/@eric_vaillancourt/using-the-python-sharepointclient-to-access-and-manage-sharepoint-files-9354361b2f9b
"""

import os

import requests


class SharePointClient:
    def __init__(self, tenant_id: str, client_id: str, client_secret: str, resource_url: str) -> None:
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.resource_url = resource_url
        self.base_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        self.headers = {"Content-Type": "application/x-www-form-urlencoded"}
        self.access_token = self.get_access_token()  # Initialize and store the access token upon instantiation

    def get_access_token(self) -> str:
        # Body for the access token request
        body = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.resource_url + ".default",
        }
        response = requests.post(self.base_url, headers=self.headers, data=body, timeout=10)
        return str(response.json().get("access_token", ""))  # Ensure the returned value is always a string

    def get_site_id(self, site_url: str) -> str:
        # Build URL to request site ID
        full_url = f"https://graph.microsoft.com/v1.0/sites/{site_url}"
        response = requests.get(full_url, headers={"Authorization": f"Bearer {self.access_token}"}, timeout=10)
        return str(response.json().get("id", ""))  # Ensure the returned value is always a string

    def get_drive_id(self, site_id: str) -> list[tuple[str, str]]:
        # Retrieve drive IDs and names associated with a site
        drives_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
        response = requests.get(drives_url, headers={"Authorization": f"Bearer {self.access_token}"}, timeout=10)
        drives = response.json().get("value", [])
        return [(drive["id"], drive["name"]) for drive in drives]

    def get_folder_content(self, site_id: str, drive_id: str, folder_path: str = "root") -> list[tuple[str, str]]:
        # Get the contents of a folder
        folder_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"
        response = requests.get(folder_url, headers={"Authorization": f"Bearer {self.access_token}"}, timeout=10)
        items_data = response.json()
        rootdir = []
        if "value" in items_data:
            for item in items_data["value"]:
                rootdir.append((item["id"], item["name"]))
        return rootdir

    # Recursive function to browse folders
    def list_folder_contents(self, site_id: str, drive_id: str, folder_id: str, level: int = 0) -> list:
        """AI is creating summary for list_folder_contents

        Args:
            site_id ([type]): [description]
            drive_id ([type]): [description]
            folder_id ([type]): [description]
            level (int, optional): [description]. Defaults to 0.

        Returns:
            list: [description]
        """
        # Get the contents of a specific folder
        folder_contents_url = (
            f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{folder_id}/children"
        )
        contents_headers = {"Authorization": f"Bearer {self.access_token}"}
        contents_response = requests.get(folder_contents_url, headers=contents_headers, timeout=10)
        folder_contents = contents_response.json()

        items_list = []  # List to store information

        if "value" in folder_contents:
            for item in folder_contents["value"]:
                if "folder" in item:
                    # Add folder to list
                    items_list.append({"name": item["name"], "type": "Folder", "mimeType": None})
                    # Recursive call for subfolders
                    items_list.extend(self.list_folder_contents(site_id, drive_id, item["id"], level + 1))
                elif "file" in item:
                    # Add file to the list with its mimeType
                    items_list.append({"name": item["name"], "type": "File", "mimeType": item["file"]["mimeType"]})

        return items_list

    def download_file(self, download_url: str, local_path: str, file_name: str) -> None:
        """AI is creating summary for download_file

        Args:
            download_url ([type]): [description]
            local_path ([type]): [description]
            file_name ([type]): [description]
        """
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(download_url, headers=headers, timeout=10)
        if response.status_code == 200:
            full_path = os.path.join(local_path, file_name)
            with open(full_path, "wb") as file:
                file.write(response.content)
            print(f"File downloaded: {full_path}")
        else:
            print(f"Failed to download {file_name}: {response.status_code} - {response.reason}")

    def download_folder_contents(
        self, site_id: str, drive_id: str, folder_id: str, local_folder_path: str, level: int = 0
    ) -> None:
        """AI is creating summary for download_folder_contents

        Args:
            site_id ([type]): [description]
            drive_id ([type]): [description]
            folder_id ([type]): [description]
            local_folder_path ([type]): [description]
            level (int, optional): [description]. Defaults to 0.
        """
        # Recursively download all contents from a folder
        folder_contents_url = (
            f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{folder_id}/children"
        )
        contents_headers = {"Authorization": f"Bearer {self.access_token}"}
        contents_response = requests.get(folder_contents_url, headers=contents_headers, timeout=10)
        folder_contents = contents_response.json()

        if "value" in folder_contents:
            for item in folder_contents["value"]:
                if "folder" in item:
                    new_path = os.path.join(local_folder_path, item["name"])
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    self.download_folder_contents(
                        site_id, drive_id, item["id"], new_path, level + 1
                    )  # Recursive call for subfolders
                elif "file" in item:
                    file_name = item["name"]
                    file_download_url = f"{resource}/v1.0/sites/{site_id}/drives/{drive_id}/items/{item['id']}/content"
                    self.download_file(file_download_url, local_folder_path, file_name)


tenant_id = os.getenv("MS_SHAREPOINT_CLIENT_TENANT_ID")
client_id = os.getenv("MS_SHAREPOINT_CLIENT_ID")
client_secret = os.getenv("MS_SHAREPOINT_CLIENT_SECRET")
site_url = os.getenv("MS_SHAREPOINT_CLIENT_SITE")
resource = os.getenv("RESOURCE")

# Usage of the class
resource = "https://graph.microsoft.com/"

client = SharePointClient(tenant_id, client_id, client_secret, resource)
site_id = client.get_site_id(site_url)
print("Site ID:", site_id)

drive_info = client.get_drive_id(site_id)
print("Root folder:", drive_info)

drive_id = drive_info[0][0]  # Assume the first drive ID
folder_content = client.get_folder_content(site_id, drive_id)
print("Root Content:", folder_content)

folder_id = folder_content[0][0]

contents = client.list_folder_contents(site_id, drive_id, folder_id)
for content in contents:
    print(f"Name: {content['name']}, Type: {content['type']}, MimeType: {content.get('mimeType', 'N/A')}")

local_save_path = "data"
client.download_folder_contents(site_id, drive_id, folder_id, local_save_path)
