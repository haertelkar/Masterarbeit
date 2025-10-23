import os
import shutil
from tqdm import tqdm


# --- Configuration ---
SOURCE_FOLDER = "measurements_test_1710_4to8s_-50def_15B_860Z_OSA_2"  # Folder A
DESTINATION_FOLDER = "measurements_test_1710_4to8s_-50def_15B_860Z_OSA" # Folder B
# ---------------------

def move_folder_contents_with_progress(source, destination):
    """
    Moves all files and subdirectories from 'source' to 'destination'
    and displays a progress bar.
    """
    # 1. Get a list of all items (files and folders) in the source directory
    items_to_move = os.listdir(source)
    
    # 2. Use tqdm to iterate over the items and show progress
    print(f"Moving contents from '{source}' to '{destination}'...")
    
    # The total will be the number of items to move
    for item in tqdm(items_to_move, desc="Progress"):
        source_path = os.path.join(source, item)
        destination_path = os.path.join(destination, item)

        if os.path.exists(destination_path):          
            # Separate the base name (filename without extension) and the extension
            base, ext = os.path.splitext(destination_path)
            destination_path = f"{base}_2{ext}"

        try:           
            shutil.move(source_path, destination_path)

        except Exception as e:
            # Print an error but continue with the next item
            print(f"\n[ERROR] Could not move '{item}': {e}")
            
    print("\n✅ Move complete!")

# --- Execution ---
if __name__ == "__main__":    
    move_folder_contents_with_progress(SOURCE_FOLDER, DESTINATION_FOLDER)