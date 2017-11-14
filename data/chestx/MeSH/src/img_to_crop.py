import os
from shutil import copyfile, move

rootdir = '/Users/elik/Documents/EliksDocs/thsis_files/learning-to-read-master/NLMCXR_dcm'
folder_1001 = '/Users/elik/Documents/EliksDocs/thsis_files/learning-to-read-master/dicom_files'
# exception_folder = '/Users/elik/Development/ex_files_xray.txt'

# with open(exception_folder) as f:
#     lines = f.readlines()
#     for line in lines:
#         try:
#             line = line.replace('\n', '');
#             move(os.path.join(folder_2001, line), os.path.join(folder_1001, line))
#             line2 = line.replace('2001', '1001')
#             move(os.path.join(folder_1001, line2), os.path.join(folder_2001, line2))
#         except IOError as e:
#             print e


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if '1001' in file:
            copyfile(os.path.join(subdir, file), os.path.join(folder_1001, file))
