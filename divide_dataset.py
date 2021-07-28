import random

txt_tables_T = []
txt_tables_F = []

with open('C:/Users/user/Desktop/label.txt','r') as f:
    for line in f:
        if "F" in line:
            txt_tables_F.append(list(line.strip('\n').split(',')))
        else:
            txt_tables_T.append(list(line.strip('\n').split(',')))
    f.close
    
#print(txt_tables_T)
#print(txt_tables_F)

eval = random.sample(txt_tables_T, 2005) + random.sample(txt_tables_F, 2005)
all = txt_tables_T + txt_tables_F
train = [i for i in all if i not in eval]
print(train)

for i in eval:
    with open("C:/Users/user/Desktop/eval.txt", "a") as file:
        file.write(str(i).lstrip("['").rstrip("']") + "\n")
        #print(i)
    file.close()
    
for i in train:
    with open("C:/Users/user/Desktop/train.txt", "a") as file:
        file.write(str(i).lstrip("['").rstrip("']") + "\n")
        print(i)
    file.close()