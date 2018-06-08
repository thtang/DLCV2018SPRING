# Action recognition
This folder contains the implementation of

* Trimmed action recognition using RNN
* Temporal action segmentation using Seq2seq model

Dataset:
* Task 1 & Task2: 4151 trimmed videos (each 5-20 secs in 24 fps with size 240x320)
* Task 3: 29 full-length videos (with size 240x320)
* 11 action labels

For details, refers to the [PPT](https://github.com/thtang/DLCV2018SPRING/blob/master/hw5/dlcv_hw5.pdf) provided by TA.

## Usage
### Training:
Refers to the [**train**](https://github.com/thtang/DLCV2018SPRING/tree/master/hw5/train) folder.


### Testing:
```bash
# Extract CNN-based feature and conduct prediction using average-pooled features
bash hw5_p1.sh [directory of trimmed validation videos folder] [path of ground-truth csv file] [directory of output labels folder]

# Extract CNN-based feature and conduct prediction through RNN
bash hw5_p2.sh [directory of trimmed validation videos folder] [path of ground-truth csv file] [directory of output labels folder]

# Whole video length action recognition
bash hw5_p3.sh [directory of full-length validation videos folder] [directory of output labels folder]
```
#### Dependency:
`Python3` `pytorch==0.4` `torchvision==0.2.1` `skimage` `matplotlib` `skvideo`

## Results:
* Performance in accuracy

|         |CNN-based feautres           | RNN-based feautres  | Temporal action prediction
| ------------- |:-------------:|:-----:|:-----:|
| *Validation Acc.*    | 0.475 | 0.510 | 0.5779

* Visualization of CNN-based video features (**left**) and RNN-based video features (**right**).

<img src="https://github.com/thtang/DLCV2018SPRING/blob/master/hw5/images/CNN_tsne.png" width=430><img src="https://github.com/thtang/DLCV2018SPRING/blob/master/hw5/images/RNN_tsne.png" width=430><br>
* Visualization of Temporal action segmentation (OP06-R05-Cheeseburger).<br>
<img src="https://github.com/thtang/DLCV2018SPRING/blob/master/hw5/images/temporal_action_segmentation_with_frames.png">

* Color index and its corresponding gener:

|     Index   |0 | 1 | 2|3|4|5|6|7|8|9|10
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| *genres*    |Other | Inspect/Read |Open|Take|Cut|Put|Close|Move Around|Divide/Pull|Pour|Transfer

## Reference:
[1] https://zhuanlan.zhihu.com/p/34418001 <br>
[2] https://github.com/thtang/ADLxMLDS2017/tree/master/hw1
