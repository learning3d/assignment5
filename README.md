## Overview
In this assignment, you will implement a PointNet based architecture for classification and segmentation with point clouds (you don't need to worry about the tranformation blocks). Q1 and Q2 focus on implementing, training and testing models. Q3 asks you to quantitatively analyze model robustness. Q4 (extra point) involves locality. 

`models.py` is where you will define model structures. `train.py` loads data, trains models, logs trajectories and saves checkpoints. `eval_cls.py` and `eval_seg.py` contain script to evaluate model accuracy and visualize segmentation result. Feel free to modify any file as needed.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Q1. Classification Model (40 points)](#q1-classification-model-40-points)
- [Q2. Segmentation Model (40 points)](#q2-segmentation-model-40-points)
- [Q3. Robustness Analysis (20 points)](#q3-robustness-analysis-20-points)
- [Q4. Bonus Question - Locality (20 points)](#q4-bonus-question---locality-20-points)

## Environment Setup
You should be able to use the environment used for previous assignments. If you need to create a new environment, please follow the set up instruction at [Assignment1](https://github.com/learning3d/assignment1). We include the same requirements.txt as A1 for your convenience.

## Data Preparation
Please download and extract the dataset for this assigment. We host the dataset on huggingface [here](https://huggingface.co/datasets/learning3dvision/assignment5/tree/main).

Download the dataset using the following commands:
```
$ sudo apt install git-lfs
$ git lfs install
$ git clone https://huggingface.co/datasets/learning3dvision/assignment5
```

The `at_data.zip` file should be downloaded at the `assignment5` subdirecotry. You can unzip it with the command:
```
$ unzip ./assignment5/a5_data.zip -d ./
```
This will produce the unzipped `data` folder under root directory. There are two folders (`cls` and `seg`) corresponding to two tasks, each of which contains `.npy` files for training and testing.



## Q1. Classification Model (40 points)
Implement the classification model in `models.py`.

- Input: points clouds from across 3 classes (chairs, vases and lamps objects)

- Output: propapibility distribution indicating predicted classification (Dimension: Batch * Number of Classes)

Complete model initialization and prediction in `train.py` and `eval_cls.py`. Run `python train.py --task cls` to train the model, and `python eval_cls.py` for evaluation. Check out the arguments and feel free to modify them as you want.

Deliverables: On your website, 

- Report the test accuracy.

- Visualize a few random test point clouds and mention the predicted classes for each. Also visualize at least 1 failure prediction for each class (chair, vase and lamp),  and provide interpretation in a few sentences.  

## Q2. Segmentation Model (40 points) 
Implement the segmentation model in `models.py`.  

- Input: points of chair objects (6 semantic segmentation classes) 

- Output: segmentation of points (Dimension: Batch * Number of Points per Object * Number of Segmentation Classes)

Complete model initialization and prediction in `train.py` and `eval_seg.py`. Run `python train.py --task seg` to train the model. Running `python eval_seg.py` will save two gif's, one for ground truth and the other for model prediction. Check out the arguments and feel free to modify them as you want. In particular, you may want to specify `--i` and `--load_checkpoint` arguments in `eval_seg.py` to use your desired model checkpoint on a particular object.

Deliverables: On your website 

- Report the test accuracy.

- Visualize segmentation results of at least 5 objects (including 2 bad predictions) with corresponding ground truth, report the prediction accuracy for each object, and provide interpretation in a few sentences.
  
## Q3. Robustness Analysis (20 points) 
Conduct 2 experiments to analyze the robustness of your learned model. Some possible suggestions are:
1. You can rotate the input point clouds by certain degrees and report how much the accuracy falls
2. You can input a different number of points points per object (modify `--num_points` when evaluating models in `eval_cls.py` and `eval_seg.py`)

Feel free to try other ways of probing the robustness. Each experiment is worth 10 points.

Deliverables: On your website, for each experiment

- Describe your procedure 
- For each task, report test accuracy and visualization on a few samples, in comparison with your results from Q1 & Q2.
- Provide some interpretation in a few sentences.

## Q4. Bonus Question - Locality (20 points)
Incorporate certain kind of locality as covered in the lecture (e.g. implement PointNet++, DGCNN, Transformers (https://arxiv.org/pdf/2012.09164v2.pdf), etc).

Deliverables: On your website, 

- specify the model you have implemented
- for each task, report the test accuracy of your best model, in comparison with your results from Q1 & Q2
- visualize results in comparison to ones obtained in the earlier parts
