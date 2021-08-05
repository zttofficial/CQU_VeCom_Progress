'''
Label the dataset
'''

import os

file_path = "/home/user/dataset/cut"
path_list = os.listdir(file_path) 

path_name = []
count = 0

for i in path_list:
    path_name.append(i)
    print(i)

path_name.sort()

for file_name in path_name:
    count = count+1
    if "F" in file_name:
        with open("/home/user/dataset/label.txt", "a") as file:
            file.write(file_name + " " + "0" + "\n")
            print(file_name)
        file.close()
    else:
        with open("/home/user/dataset/label.txt", "a") as file:
            file.write(file_name + " " + "1" + "\n")
            print(file_name)
        file.close()
