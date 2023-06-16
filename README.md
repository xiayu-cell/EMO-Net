# EMO-Net

### **Lightweight Network with Feature Fusion Branchand Local Attention Block for Facial Expression Recognition**

![overall](.static\overall.jpg)

A PyTorch implementation of the [EMO-Net](), pre-trained models are available in checkpoint.

___

**Local Attention Block -- Efficient Channel Attention**

![ECA](D:\study\班导任务\论文\FER\EMO-Net\static\ECA.jpg)

​			 The architecture of the ECA mechanism used in the inverted residual structure, where GAP denotes global average pooling and σ denotes the Sigmoid function.

___

**Feature Fusion Branch**

![fusion_branch](.static\fusion_branch.jpg)

​				Feature fusion branch is utilized in the inverted residual structure. Main branch is the feature extractor which combines with auxiliary branch with shallow geometry feature. DW 3x3 denotes a depth-separable convolution, and ⊕ denotes a summary of two branchs.

****

![fusion](.static\fusion.jpg)

​					The architecture of the two fusion branches in the inverted residual structure, including the main branch (feature extractor and local attention module) and the auxiliary branch

#### **Requirements**

- Python >= 3.6
- PyTorch >= 1.2
- torchvision >= 0.4.0

#### **Training**

- Step 1: download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/raf/model1.html), and make sure it have the structure like following:

  ```
  ./RAF-DB/
           train/
                 0/
                   train_09748.jpg
                   ...
                   train_12271.jpg
                 1/
                 ...
                 6/
           test/
                0/
                ...
                6/
  [Note] 0: surprise; 1: fear; 2: disgust; 3: happy; 4: sad; 5: anger; 6: neutral
  ```

- Step 2: load the model weight in the **./checkpoint**
- Step 3: change ***data_path*** in **train.py** to your path
- Step 4: run `python train.py `

#### **Model Weight**

| Dataset     | accuracy (%) | λ    | weigth.pth                   |
| ----------- | ------------ | ---- | ---------------------------- |
| RAFDB       | 91.30        | 0.5  | RAFDB_a_0.5_0.913.pth        |
| RAFDB       | 91.52        | 0.6  | rafdb_a_0.6_0.915.pth        |
| AffectNet-7 | 65.456       | 0.6  | affectNet7_a_0.6_0.65457.pth |
| AffectNet-8 | 62.08        | 0.6  | affectNet8_a_0.6_0.6208.pth  |


