# import os

# copy files with names 'best_' from one directory to another

import os
import shutil

source_directory = r"C:\master_thesis\ma_th_test\best_files_exp\exp_search_space\results"  # Specify the source directory path
target_directory = r"C:\master_thesis\ma_th_test\best_files_exp\exp_search_space\all_activity_avg_std"  # Specify the target directory path

# Iterate over the subdirectories and files in the source directory
for root, dirs, files in os.walk(source_directory):
    for file in files:
        if file.startswith('activity_avg_std'):
            source_file_path = os.path.join(root, file)
            target_file_path = os.path.join(target_directory, file)
            shutil.copy2(source_file_path, target_file_path)


# # # remove timestamp from folder names and update seed numbers if any
# folder_path = r"C:\Users\minio\Downloads\20230529 - Copy"  # Specify the folder path containing the folders

# # Iterate over the folders in the folder path
# for foldername in os.listdir(folder_path):
#     if os.path.isdir(os.path.join(folder_path, foldername)):
#         old_text = foldername
#         x = old_text.split("_")
#         x = x[1:] # removing the time stamp
#         new_foldername = '_'.join(x)
        
#         # Generate the new folder name
#         # new_foldername = foldername.replace("old_text", "new_text")  # Modify the renaming logic as needed

#         # Rename the folder
#         os.rename(os.path.join(folder_path, foldername), os.path.join(folder_path, new_foldername))

# change file names 
# import os
# folder_path = r"C:\Users\minio\Downloads\best_files_exp_search_space-Copy"  # Specify the folder path containing the files

# # Iterate over the files in the folder
# for filename in os.listdir(folder_path):
#     if os.path.isfile(os.path.join(folder_path, filename)):
#         old_text = filename
#         x = old_text.split("_")
#         x = x[:-1] + ['2']
#         new_filename = '_'.join(x) + '.csv'
#         # print(new_text)
#         # Generate the new file name
#         # new_filename = filename.replace("old_text", "new_text")  # Modify the renaming logic as needed

#         # Rename the file
#         os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

# change file names of all the files satrting with name 'best_' in a directory

# import os

# directory_path = r"C:\Users\minio\Downloads\20230515-Copy\seed_3"  # Specify the directory path

# # Iterate over the root directory and its subdirectories
# for root, dirs, files in os.walk(directory_path):
#     for filename in files:
#         if filename.startswith("results_"):
#             old_filepath = os.path.join(root, filename)
#             old_text = filename
#             temp_filename = old_text.split("_")
#             temp_filename = ['best'] + temp_filename # renaem the files as seed_3
#             new_filename = '_'.join(temp_filename) + '.csv'
#             new_filepath = os.path.join(root, new_filename) 
#             # Rename the file
#             os.rename(old_filepath, new_filepath)
