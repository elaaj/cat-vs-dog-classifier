# cat-vs-dog-classifier


## Table of Contents

- [Description](#Description)
- [Installation](#Installation)
- [Usage](#usage)


## Description:

The goal of this project is to devise an accurate CNN-based classifier able to distinguish between Cat and Dog in images where the animal is predominant. For this task, the used CNN is a pre-trained VGG11 provided by Pytorch, trained using a Weighted Adam optimizer and using specific learning rates per epoch, determined through several tests and trials, performed also thanks to standard learning rate schedulers provided by PyTorch.
The Python programs included are:
- vggTrainer.py: includes the utilities devised to train the network over the dataset.
- trainingEvaluation.py: defines the evaluation utilities, useful to evaluate the network after training sessions.
- main.py: includes the Python main, which uses all the other programs.
- classifierDemo.ipynb: Jupyter notebook used to display the capabilities of the network over the dataset.

## Installation

To run the Python training algorithm and the Demo notebook there are some requirements:

- Python 3.
- The following python modules:
  - matplotlib
  - PIL
  - torch
  - torchvision
  - albumentations
 
The Pytorch checkpoint is not included for purposes of space, but it can be easily reproduced through the given algorithm, which does run for 3 brief epochs. At the current time the algorithm is devised to run only on CUDA-GPUs, so if you want to run it on CPU you must modify the device used by PyTorch in the code.
The dataset is provided [here](https://www.kaggle.com/competitions/dog-vs-cat-classification/overview), and should be placed in a folder named **DATASET**. In addition, the original folders of the **train** and **test** datasets contain 'wrapper' folders, which have the same name as the parent folder, so I decided to remove them for simplicity's sake, and so should you, otherwise you will have to edit the main code and the demo code.

## Usage

The main notebook can be run from any IDE which supports Jupyter Notebooks, whereas to run the Python algorithms a terminal or a suitable IDE are required.

```bash
git clone https://github.com/elaaj/cat-vs-dog-classifier
```
