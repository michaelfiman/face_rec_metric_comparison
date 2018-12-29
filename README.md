# face_rec_metric_comparison
A computer vision CNN DL project to create a face recognition model, comparing several types of loss metrics for training.

## Overview
In this project different loss metrics will be compared in order to train a model which will be able to perform a face recognition objective.
The objective will be tested by comparing the distances between output vectors from a model which is given face images as inputs. It will attempt to predict if 2 images are of the same person according to the distance between.

The original loss metric which was used to do this project was a self created loss score based on the L2 distance between 2 vectors and the known fact if the images are of the same person or not (will be elaborated further below).
The other metrics compared are taken from (tf.contrib.losses.metric_learning):
1. contrastive loss
2. npairs loss
3. lifted struct loss
4. cluster loss

## Loss metrics

### Self created loss
The main idea behind this loss is to penalize "similar" images if they are far from one another and "different" images if they are too close.
