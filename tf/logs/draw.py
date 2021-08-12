from matplotlib import pyplot as plt
import re

filename = 'log4.txt'
step, loss, acc, f1 = [], [], [], []

with open(filename, 'r') as f:

    lines = f.readlines()
    j = 0
    for line in lines:
        temp = re.split(r":|\s", line)
        #print(temp)
        step.append(j)
        j = j + 1
        loss.append(float(temp[4]))
        acc.append(float(temp[7]))
        f1.append(float(temp[10]))
        

fig = plt.figure(figsize=(10, 10))
#first figure
ax1 = fig.add_subplot(3,1,1)
ax1.plot(step, loss, 'red', label='loss')  
ax1.legend(loc='upper right')  
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')  
#second figure
ax2 = fig.add_subplot(3,1,2)  
ax2.plot(step, acc, 'blue', label='acc')  
ax2.legend(loc='upper right')  
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
#third figure
ax3 = fig.add_subplot(3,1,3)  
ax3.plot(step, f1, 'green', label='f1')
ax3.legend(loc='upper right')  
ax3.set_xlabel('epoch')
ax3.set_ylabel('micro_f1')
plt.show()