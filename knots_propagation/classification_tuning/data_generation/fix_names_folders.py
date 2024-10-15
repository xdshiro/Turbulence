import os
import shutil
import re


def organize_npy_files(base_folder):
    # Walk through the base folder to find all .npy files
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.npy'):
                # Extract the directory structure from the filename
                match = re.match(r'(.+?)\\(.+)', file)  # Adjust to Windows-like pattern
                if match:
                    subdirs, filename = match.groups()
                    # Create the original folder structure if not present
                    target_folder = os.path.join(base_folder, *subdirs.split('_'))
                    os.makedirs(target_folder, exist_ok=True)

                    # Move and rename the .npy file to the correct folder
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(target_folder, filename)
                    shutil.move(old_path, new_path)
                    print(f'Moved: {old_path} -> {new_path}')


if __name__ == '__main__':
    # Replace with your base folder path
    base_folder = "/path/to/your/folder"
    organize_npy_files(base_folder)