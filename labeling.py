import os

file_path = "D:/CQU/车身对比/dataset_F"
path_list = os.listdir(file_path) 

path_name = []
count = 0

for i in path_list:
    path_name.append(i)
    print(i)

#path_name.sort()

for file_name in path_name:
    count = count+1
    with open("C:/Users/user/Desktop/dataset_F.txt", "a") as file:
        file.write(file_name + "\t" + "0" + "\n")
        print(file_name)
    file.close()
