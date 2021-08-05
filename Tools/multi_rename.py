'''
Batch rename
'''

import os
 
file_path = "D:/CQU/车身对比/dataset_F"
path_list = os.listdir(file_path) 
path_name = []
count = 0


for i in path_list:
    path_name.append(i)
    print(i)

os.chdir('D:/CQU/车身对比/dataset_F')
for old_name in path_list:
    os.rename(old_name, str(count) + "_F.jpg")
    count = count+1