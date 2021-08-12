from matplotlib import pyplot as plt
import re


filename = 'log1.txt'
step, loss, acc = [], [], []

with open(filename, 'r') as f:

    lines = f.readlines()
    j = 0
    for line in lines:
        temp = re.split(r":|\s", line)
        step.append(j)
        j = j + 1
        loss.append(float(temp[4]))
        acc.append(float(temp[-2]))
        

fig = plt.figure(figsize=(10, 10))  
#first figure
ax1 = fig.add_subplot(211)
ax1.plot(step, loss, 'red', label='loss')  
ax1.legend(loc='upper right')  
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')  
#second figure
ax2 = fig.add_subplot(212)  
ax2.plot(step, acc, 'blue', label='acc')  
ax2.legend(loc='upper right')  
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
plt.show()