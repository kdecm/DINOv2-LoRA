import os
import shutil
import random

# Define the path to your dataset directory
dataset_dir = "/home/UserData/dino/LoRA/seed_dataset/"  # Replace this with your dataset folder path

# Define the train, validation, and test directories
train_dir = 'path_to_train_dir'  # Replace with your desired train directory path
val_dir = 'path_to_val_dir'      # Replace with your desired validation directory path
test_dir = 'path_to_test_dir'    # Replace with your desired test directory path

# Create the train, validation, and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    
    if os.path.isdir(class_path):  # Ensure it's a directory
        # Create corresponding subdirectories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        # Shuffle the image files
        random.shuffle(image_files)
        
        # Calculate split indices
        total = len(image_files)
        train_end = int(0.7 * total)
        val_end = int(0.8 * total)  # 70% train, next 10% val, remaining 20% test

        # Split files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]

        # Copy files
        for file in train_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(train_dir, class_name, file))
        
        for file in val_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(val_dir, class_name, file))
        
        for file in test_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(test_dir, class_name, file))

print("Dataset split into train (70%), validation (10%), and test (20%) sets.")

