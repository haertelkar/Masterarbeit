import glob
import os
from tqdm import tqdm
import time

def delete_folders_with_nested_progress(pattern):
    """
    Finds and deletes folders matching a wildcard pattern.
    Shows a progress bar for folders and a nested bar for files inside.
    NOTE: This version assumes folders only contain files, as requested.
    """
    
    print(f"Searching for folders matching: '{pattern}'")
    
    # 1. Find all paths matching the wildcard pattern
    all_matches = glob.glob(pattern)
    
    # 2. Filter the list to include only directories
    folders_to_delete = [path for path in all_matches if os.path.isdir(path)]
    
    if not folders_to_delete:
        print("No matching folders found. Exiting.")
        return

    # 3. CRITICAL: Show the user what will be deleted and ask for confirmation
    print("\n--- Folders Found ---")
    for folder in folders_to_delete:
        print(f"  [+] {folder}")
    print("----------------------")
    
    print(f"\nFound {len(folders_to_delete)} folders to delete.")
    
    try:
        confirm = input("Are you ABSOLUTELY SURE you want to delete all these folders and their contents? (yes/no): ")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return

    if confirm.lower().strip() != 'yes':
        print("Operation cancelled. No files were deleted.")
        return

    print("\nStarting deletion...")
    
    # 4. Outer loop: Iterate through each FOLDER with a progress bar
    try:
        # Use position=0 for the main (outer) progress bar
        for folder in tqdm(folders_to_delete, desc="Total Progress (Folders)", unit="folder", position=0):
            
            # Find all files *inside* this specific folder
            # We create a list first to know the total count for the progress bar
            files_to_delete = []
            for f in os.listdir(folder):
                file_path = os.path.join(folder, f)
                if os.path.isfile(file_path):
                    files_to_delete.append(file_path)

            if files_to_delete:
                # 5. Inner loop: Iterate through each FILE with a nested progress bar
                # Use position=1 to nest it visually below the main bar
                # Use leave=False so this bar disappears after it's done
                folder_name = os.path.basename(folder)
                for file in tqdm(files_to_delete, desc=f"  Deleting in {folder_name[:20]:<20}", unit="file", position=1, leave=False):
                    try:
                        os.remove(file)
                        # time.sleep(0.01) # Slow down to see the bar (optional)
                    except OSError as e:
                        print(f"\nError: Could not delete file {file}. Reason: {e}")
            
            # 6. After all files are gone, delete the now-empty folder
            try:
                os.rmdir(folder)
            except OSError as e:
                print(f"\nError: Could not delete empty folder {folder}. Reason: {e}")

        # Clear the space for the nested bar
        print("\n" * 2) 
        print("✅ Deletion complete.")

    except KeyboardInterrupt:
        print("\nDeletion stopped mid-process by user.")


# --- Main execution ---
if __name__ == "__main__":
    try:
        # Get the pattern from the user
        user_pattern = input("Enter wildcard pattern for folders (e.g., *temp* or build_*): ")
        
        if not user_pattern:
            print("No pattern entered. Exiting.")
        else:
            delete_folders_with_nested_progress(user_pattern)
            
    except KeyboardInterrupt:
        print("\nScript terminated by user.")