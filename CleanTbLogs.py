import os
import shutil
import time

def get_folder_size(folder):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def remove_small_folders(path, size_limit_kb):
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                folder_path = entry.path
                folder_size = get_folder_size(folder_path) / 1024  # Convert to KB
                if folder_size < size_limit_kb:
                    response = input(f"Do you want to remove the folder '{folder_path}' of size {folder_size:.2f} KB? (y/n): ")
                    if response.lower() == 'y':
                        try:
                            shutil.rmtree(folder_path)
                        except OSError:
                            time.sleep(2)
                            shutil.rmtree(folder_path)
                        print(f"Removed folder: {folder_path}")

if __name__ == "__main__":
    directory_path = "tb_logs"
    remove_small_folders(directory_path, 20)