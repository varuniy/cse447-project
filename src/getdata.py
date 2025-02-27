import requests
import tarfile
import re
from bs4 import BeautifulSoup
import os
import random

# Step 1: Send a request to the URL and get the HTML content
url = "https://wortschatz.uni-leipzig.de/en/download"
response = requests.get(url)

# Step 2: Check if the request was successful
if response.status_code == 200:
    print("Successfully accessed the page.")
    
    # Step 3: Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Step 4: Find all elements with the class 'list-group-item'
    list_items = soup.find_all('li', class_='list-group-item')
    
    # Step 5: Extract the 'href' attribute from each link
    for item in list_items:
        link = item.find('a')  # Find the <a> tag within the <li>
        if link and link.get('href'):
            suffix = link.get('href').split('/')[-1]  # Extract the suffix (e.g., "ven")
            download_url = f"https://downloads.wortschatz-leipzig.de/corpora/{suffix}_wikipedia_2021_30K.tar.gz"  # Build the full download URL
            print(f"Found download link: {download_url}")
            
            # Step 6: Download the file
            file_response = requests.get(download_url)
            
            # Step 7: Save the file locally in the 'data' directory outside the current directory
            if file_response.status_code == 200:
                filename = f"{suffix}_wikipedia_2021_30K.tar.gz"  # Use the suffix to name the file
                data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data')  # Go one level up from current directory and add 'data'
                os.makedirs(data_dir, exist_ok=True)  # Create the directory if it doesn't exist
                
                file_path = os.path.join(data_dir, filename)  # Full path to save the file
                
                with open(file_path, 'wb') as f:
                    f.write(file_response.content)
                print(f"File '{filename}' downloaded successfully to '{file_path}'.")
            else:
                print(f"Failed to download the file from {download_url}.")
else:
    print("Failed to retrieve the page.")

# Define the paths
src_dir = os.getcwd()  # current working directory (in 'src' directory)
data_dir = os.path.join(src_dir, '..', 'data')  # path to the 'data' folder outside 'src'

# List all .tar.gz files in the 'data' directory
tar_files = [f for f in os.listdir(data_dir) if f.endswith('.tar.gz')]

# Iterate over each .tar.gz file
for tar_file in tar_files:
    tar_file_path = os.path.join(data_dir, tar_file)
    
    # Extract the file prefix (the part before .tar.gz)
    file_prefix = tar_file.replace('.tar.gz', '')
    
    # Open the .tar.gz file
    with tarfile.open(tar_file_path, 'r:gz') as tar:
        # Check for the 'file_prefix' folder and the file inside it
        folder_name = file_prefix
        sentences_file = f"{file_prefix}-sentences.txt"
        
        # Loop through all members in the archive to find the file
        for member in tar.getmembers():
            # Check if the member is the correct file inside the folder
            if member.name.endswith(sentences_file) and member.name.startswith(folder_name):
                # We want to extract only the file without its folder structure
                extracted_file_path = os.path.join(data_dir, sentences_file)
                
                # Extract the file to the 'data' folder with the correct name
                with open(extracted_file_path, 'wb') as f_out:
                    f_out.write(tar.extractfile(member).read())
                print(f"Extracted: {sentences_file} from {tar_file}")
                break
        else:
            print(f"File not found: {sentences_file} in folder {folder_name} of {tar_file}")

print("Extraction complete.")

# Directory where your .txt files are located
directory = "../data/"  # Adjust this path

# Initialize an empty string to store all the cleaned content
combined_text = ""

# Define the regular expression pattern to remove the numbering at the start of each line
pattern = r'^\d+\s*;?\s*'  # Matches the number and optional semicolon, followed by optional spaces

# Initialize the variables to store the training and validation text
training_text = ""
validation_text = ""

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # Ensure you're reading .txt files
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
            lines = f.readlines()  # Read all lines from the file
            
            # Ensure the file has exactly 30,000 lines as per the given assumption
            if len(lines) == 30000:
                # Shuffle the lines randomly
                random.shuffle(lines)
                
                # Separate the first 27,000 lines for training and the last 3,000 for validation
                training_lines = lines[:27000]
                validation_lines = lines[27000:]
                
                # Clean and append the training lines to training_text
                for line in training_lines:
                    cleaned_line = re.sub(pattern, '', line).strip()  # Remove unwanted part
                    if cleaned_line:  # Avoid appending empty lines
                        training_text += cleaned_line + "\n"
                
                # Clean and append the validation lines to validation_text
                for line in validation_lines:
                    cleaned_line = re.sub(pattern, '', line).strip()  # Remove unwanted part
                    if cleaned_line:  # Avoid appending empty lines
                        validation_text += cleaned_line + "\n"
            else:
                print(f"Warning: File {filename} does not have 30,000 lines.")

# Now save the cleaned and combined text to training.txt and validation.txt
with open('../data/training.txt', 'w', encoding='utf-8') as f:
    f.write(training_text)

with open('../data/validation.txt', 'w', encoding='utf-8') as f:
    f.write(validation_text)


