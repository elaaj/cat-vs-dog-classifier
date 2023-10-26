# cat_vs_dog_classifier


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

To run the Python algorithms ther

- Python 3.
- The following python modules:
  - matplotlib
  - opencv-python
  - numpy
  - sklearn
 
Also, the "data" folder must be placed outside of the folder containing all the algorithms, or the path in the main file must be adapted to the new data location.

## Usage

The main notebook can be run from any IDE which supports Jupyter Notebooks, whereas to run the Python algorithms a terminal or a suitable IDE are required.



```bash
git clone https://github.com/elaaj/cat-vs-dog-classifier
```
