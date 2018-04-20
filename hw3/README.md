# Sementic Segmentation
<img src="https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/flow.png" width="500">

## Usage
#### Preprocessing
Refers to [preprocessing.ipynb](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/preprocessing.ipynb).
#### Training
Refers to [fcn32.ipynb](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/fcn32.ipynb).

#### Testing
```
python3 fcn32_inference.py [GPU id] [model weight path] [input dir] [output dir]
```

## Baseline Performance
#### FCN32s
```
class #0 : 0.74382
class #1 : 0.88017
class #2 : 0.35570
class #3 : 0.78427
class #4 : 0.72560
class #5 : 0.70210
mean iou score: 0.698
```
<img src="https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/legend.png" width="320">

Satellite       |  Ground truth | Prediction
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/sat.png)  |  ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/ground_truth.png) | ![](https://github.com/thtang/DLCV2018SPRING/blob/master/hw3/image/prediction.png)

## Reference
TA's PDF
