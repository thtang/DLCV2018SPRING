# Sementic Segmentation
<img src="https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/flow.png" width="500">

## Usage
#### Preprocessing
Refers to [preprocessing.ipynb](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/preprocessing.ipynb).
#### Training
Refers to [train.ipynb](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/train.ipynb).

#### Testing
```
python3 fcn32_inference.py [GPU id] [model weight path] [input dir] [output dir]
```

## Baseline Performance
class|  FCN32s | FCN8s 
:-------------------------:|:-------------------------:|:-------------------------:
class 0|0.70796|0.75347
class 1|0.87159 |0.88461
class 2| 0.28390 |0.34221
class 3| 0.78811|0.79009
class 4| 0.69469|0.73988
class 5| 0.67908|0.68888
mean_iou | **0.670888** | **0.699856**


<img src="https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/legend.png" width="320">

FCN32s results:

 Validation img|Satellite       |  Ground truth | Epoch 1 |Epoch 10 | Epoch 20
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
0008|![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0008_sat.jpg)  |  ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0008_mask.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0008_mask_1_o.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0008_mask_10_o.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0008_mask_20_o.png) 
0097|![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0097_sat.jpg)  |  ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0097_mask.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0097_mask_1_o.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0097_mask_10_o.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0097_mask_20_o.png) 
0107|![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0107_sat.jpg)  |  ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0107_mask.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0107_mask_1_o.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0107_mask_10_o.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0107_mask_20_o.png) 

FCN8s results:

 Validation img|Satellite       |  Ground truth | Epoch 1 |Epoch 10 | Epoch 20
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
0008|![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0008_sat.jpg)  |  ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0008_mask.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0008_mask_1.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0008_mask_10.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0008_mask_20.png) 
0097|![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0097_sat.jpg)  |  ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0097_mask.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0097_mask_1.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0097_mask_10.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0097_mask_20.png) 
0107|![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0107_sat.jpg)  |  ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0107_mask.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0107_mask_1.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0107_mask_10.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/0107_mask_20.png) 
## Reference
TA's PDF [[1]](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/DLCV_hw3.pdf)<br>
Fully Convolutional Networks for Semantic Segmentation [[2]](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
