## Image processing method 1

------

Connect two pictures horizontally.



###ResNet50_vd

| Train | Eval |
| ----- | ---- |
| 29760 | 7440 |

![image-20210723135948249](.\images\image-20210723135948249.png)

![image-20210723140036530](.\images\image-20210723140036530.png)

![image-20210723140057898](.\images\image-20210723140057898.png)

![image-20210723140114163](.\images\image-20210723140114163.png)

![image-20210723140133839](.\images\image-20210723140133839.png)

![image-20210723140149188](.\images\image-20210723140149188.png)

```python
Bestmodel:
[2021/07/23 14:25:14] root INFO: [Eval][Epoch 0][Avg]CELoss: 0.70945, loss: 0.70945, top1: 0.47661, top5: 1.00000
Latest:
[2021/07/23 14:58:37] root INFO: [Eval][Epoch 0][Avg]CELoss: 1.45308, loss: 1.45308, top1: 0.20860, top5: 1.00000
```

### EfficientNetB0

#### Round 1

* `Metric` configuration is wrong ( with a binary classification problem, its `topk` should be [1,1] ), `class_num` is defined in the `Global`.

| Train | Eval |
| ----- | ---- |
| 29760 | 7440 |

![image-20210723232724336](.\images\image-20210723232724336.png)

![image-20210723233347869](.\images\image-20210723233347869.png)

![image-20210723232815994](.\images\image-20210723232815994.png)

![image-20210723232833450](.\images\image-20210723232833450.png)

#### Round 2 (fixed)

![image-20210725021250312](.\images\image-20210725021250312.png)

![image-20210725021326679](.\images\image-20210725021326679.png)

![image-20210725021400049](.\images\image-20210725021400049.png)

### AlexNet

![image-20210724205152493](.\images\image-20210724205152493.png)

#### ISSUES

```tex
Loss: NaN
```



##Image processing method 2

Use YOLO to select the main body of the car in advance, and form a new data set based on this.

| Train | Eval |
| ----- | ---- |
| 24019 | 6005 |

### EfficientNetB0

#### ![image-20210727145817732](.\images\image-20210727145817732.png)

![image-20210727145848989](.\images\image-20210727145848989.png)

![image-20210727145909609](.\images\image-20210727145909609.png)

###Image processing method 3

------

Combine two RGB pictures into a six-channel matrix, export as `.npy` format.

Then use the `tifffile` library to convert the `.npy` file to a picture in `.tif` format, which will also be divided into two types, one is `minisblack`, and the other is `rgb`.

| Train | Eval |
| ----- | ---- |
| 16042 | 4010 |

![image-20210728231204642](.\images\image-20210728231204642.png)

![image-20210728231258699](.\images\image-20210728231258699.png)

| Train | Eval |
| ----- | ---- |
| 16040 | 4010 |

*(remove object without attribute "shape")*



### Issues

+ Occupies a lot of memory
+ Customize dataset, modify dataloader

