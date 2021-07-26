from PIL import Image
import os
import random

file_path1 = "D:/CQU/车身对比/车身对比0330-0430_1"
file_path2 = "D:/CQU/车身对比/0330-0430"
path_list1 = os.listdir(file_path1) 
path_list2 = os.listdir(file_path2)

path_name1 = []
path_name2 = []
count = 0
 
def pic_joint(path1, path2, num, flag='horizontal'):
    """
    :param path1: path
    :param path2: path
    :param flag: horizontal or vertical
    :return:
    """  
    img1,img2 = Image.open(path1), Image.open(path2)

    img1 = img1.resize((1440, 1080), Image.ANTIALIAS)
    img2 = img2.resize((1440, 1080), Image.ANTIALIAS)
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
   
        joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)

        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save("D:/CQU/车身对比/splicing_F/"+str(num+18600)+"_F.jpg")
        
    elif flag == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1] + size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save("D:/CQU/车身对比/splicing_F/"+str(num+18600)+"_F.jpg")
        
for i in path_list1:
    path_name1.append(i)
    #print(i)

for j in path_list2:   
    path_name2.append(j)
    print(j)
    
for file_name in path_name1:   
    path1 = file_path1 + "/" + path_name1[count]
    #path2 = file_path2 + "/" + path_name2[count]
    path2 = file_path2 + "/" + path_name2[random.choice(list(range(0,count)) + list(range(count+1, 15013)))]
    pic_joint(path1, path2, count, flag='horizontal')
    count = count + 1
    print(count)