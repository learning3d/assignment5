## Overview
In this assignment, you will implement a PointNet based architecture for classification and segmentation with point clouds (you don't need to worry about the tranformation blocks). Q1 and Q2 focus on implementing, training and testing models. Q3 asks you to quantitatively analyze model robustness. Q4 (bonus) involves locality. 

`models.py` is where you will define model structures. `train.py` loads data, trains models, logs trajectories and saves checkpoints. `eval_cls.py` and `eval_seg.py` contain script to evaluate model accuracy and visualize segmentation result. Feel free to modify any file as needed.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Q1. Classification Model (30 points)](#q1-classification-model-30-points)
- [Q2. Segmentation Model (30 points)](#q2-segmentation-model-30-points)
- [Q3. Robustness Analysis (20 points)](#q3-robustness-analysis-20-points)
- [Q4. Bonus Question - Locality (20 points)](#q4-bonus-question---locality-20-points)

## Environment Setup
You should be able to use the environment used for previous assignments. If you need to create a new environment, please follow the set up instruction at [Assignment1](https://github.com/learning3d/assignment1). We include the same requirements.txt as A1 for your convenience.

## Data Preparation
Please download and extract the dataset for this assigment. We host the dataset on huggingface [here](https://huggingface.co/datasets/learning3dvision/assignment5/tree/main).

Download the dataset from the repository root:

```
$ pip install huggingface_hub
$ hf download learning3dvision/assignment5 a5_data.zip --repo-type dataset --local-dir data_download
$ unzip data_download/a5_data.zip -d ./
```

This writes a `data` folder at the repository root with two subfolders (`cls` and `seg`), each containing `.npy` files for training and testing.



## Q1. Classification Model (30 points)
Implement the classification model in `models.py`.

- Input: points clouds from across 3 classes (chairs, vases and lamps objects)

- Output: propapibility distribution indicating predicted classification (Dimension: Batch * Number of Classes)

Complete model initialization and prediction in `train.py` and `eval_cls.py`. Run `python train.py --task cls` to train the model, and `python eval_cls.py` for evaluation (loads `checkpoints/cls/best_model.pt` by default). Check out the arguments and feel free to modify them as you want.

Deliverables: On your website, 

- Report the test accuracy.

- Visualize a few random test point clouds and mention the predicted classes for each. 

- Also visualize at least 1 failure prediction for each class (chair, vase and lamp),  and provide an interpretation of the failure in a few sentences.  

## Q2. Segmentation Model (30 points) 
Implement the segmentation model in `models.py`.  

- Input: points of chair objects (6 semantic segmentation classes) 

- Output: segmentation of points (Dimension: Batch * Number of Points per Object * Number of Segmentation Classes)

Complete model initialization and prediction in `train.py` and `eval_seg.py`. Run `python train.py --task seg` to train the model. Running `python eval_seg.py` will save two gif's, one for ground truth and the other for model prediction (loads `checkpoints/seg/best_model.pt` by default). Check out the arguments and feel free to modify them as you want. In particular, you may want to specify `--i` and `--load_checkpoint` arguments in `eval_seg.py` to use your desired model checkpoint on a particular object.

**Deliverables**: On your website,

- Report the test accuracy.

- Visualize segmentation results of at least 5 objects (including 2 bad predictions) with corresponding ground truth

- Report the prediction accuracy for each object, and provide an interpretation of the result in a few sentences.
  
## Q3. Robustness Analysis (20 points) 
Conduct 2 experiments to analyze the robustness of your learned model. Each experiment is worth 10 points. A maximum of 20 points is possible for this question. Some possible suggestions are:
1. You can rotate the input point clouds by certain degrees and report how much the accuracy falls
2. You can input a different number of points points per object (modify `--num_points` when evaluating models in `eval_cls.py` and `eval_seg.py`)

Each experiment should be done on both the classification and segmentation tasks. Please also feel free to try other ways of probing the robustness. 

**Deliverables**: On your website, for **each experiment**,

- Briefly describe your procedure 
- For each task (classification and segmentation), report test accuracy and visualization on a few samples, in comparison with your results from Q1 & Q2.
- Provide some interpretation of the results in a few sentences.

## Q4. Bonus Question - Locality (20 points)
Incorporate certain kind of locality as covered in the lecture (e.g. implement PointNet++, DGCNN, Transformers (https://arxiv.org/pdf/2012.09164v2.pdf), etc).

**Deliverables**: On your website, 

- Specify and briefly describe the model you have implemented
- For each task (classification and segmentation), report the test accuracy of your best model, in comparison with your results from Q1 & Q2.
- Visualize some results in comparison to ones obtained in the earlier questions
