# Action recognition
This folder contains the implementation of

* Trimmed action recognition using RNN
* Temporal action segmentation using Seq2seq model

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
