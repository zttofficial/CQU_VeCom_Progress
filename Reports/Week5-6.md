Change the framework to TensorFlow.

### Alex

####TensorBoard Test (train):

![image-20210808010239732](.\images\image-20210808010239732.png)

![image-20210808010255412](.\images\image-20210808010255412.png)

#### Issues:

Because `model.fit` is in a loop in this method, multiple epochs can only be connected with time as the abscissa, and the display effect is as shown in the figure above.

Possible Solution: Draw points without visualization tool.

------

#### Eval Round 1

```
./models/1_epochs.h5   loss:0.44866931438446045  acc:0.831944465637207
./models/2_epochs.h5   loss:1.1725497245788574  acc:0.8152777552604675
./models/3_epochs.h5   loss:0.9772260189056396  acc:0.7256944179534912
./models/4_epochs.h5   loss:0.5952962040901184  acc:0.8090277910232544
./models/5_epochs.h5   loss:0.8489556908607483  acc:0.7541666626930237
./models/6_epochs.h5   loss:0.8438572287559509  acc:0.7993055582046509
./models/7_epochs.h5   loss:1.0684117078781128  acc:0.8131944537162781
./models/8_epochs.h5   loss:1.4724829196929932  acc:0.7395833134651184
./models/9_epochs.h5   loss:1.0527535676956177  acc:0.8034722208976746
./models/10_epochs.h5   loss:0.9983411431312561  acc:0.7958333492279053
```

#### Issues:

The validation set of the first epoch has the highest accuracy.

Increase the number of epochs.

------

#### Eval Round 2

![image-20210811175812068](.\images\image-20210811175812068.png)

#### Issues:

A mistake was made in the division of the dataset. 

------

#### Eval Round 3

|          | Train | Test |
| -------- | ----- | ---- |
| Positive | 29720 | 3893 |
| Negative | 29045 | 3904 |
| Total    | 58765 | 7797 |

![image-20210812110104891](.\images\image-20210812110104891.png)

#### Issues:

Analyze that overfitting after the tenth epoch causes loss to rise.

Add other more suitable metrics like F1 Score in subsequent experiments.

------

#### Eval Round 4

![image-20210812230230350](.\images\image-20210812230230350.png)

------

#### Eval Round 5 & 6

Fix some obvious inappropriate places in the code: 

Changed the loss function from `sparse_categorical_crossentropy` to `binary_crossentropy`.

Changed the output of the last layer of the network from `Dense(2, activation='softmax')` to `Dense(1, activation='sigmoid')`.

##### 5:

![image-20210813005804405](.\images\image-20210813005804405.png)

##### 6:

![image-20210813021332316](.\images\image-20210813021332316.png)

------

####8 epochs circumstance:

![image-20210813223326662](.\images\image-20210813223326662.png)

