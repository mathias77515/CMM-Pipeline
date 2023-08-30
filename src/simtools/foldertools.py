import os

def create_folder_if_not_exists(folder_name):
    # Check if the folder exists
    if not os.path.exists(folder_name):
        try:
            # Create the folder if it doesn't exist
            os.makedirs(folder_name)
            print(f"The folder '{folder_name}' has been created.")
        except OSError as e:
            print(f"Error creating the folder '{folder_name}': {e}")
    else:
        pass
