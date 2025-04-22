import gdown
import os

# --- CONFIG ---
folder_id = "1KNxZ2vwk1KqSGgCysPA8puiXvPnlVqxN"  # <-- put your folder ID here
output_dir = "data"  # Folder to save the downloaded content

# --- MAKE OUTPUT DIRECTORY ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- DOWNLOAD FROM DRIVE ---
print(f"Downloading folder from Google Drive (ID: {folder_id})...")
gdown.download_folder(
    id=folder_id,
    output=output_dir,
    quiet=False,
    use_cookies=False
)

print("✅ Download complete!")
