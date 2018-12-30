# face_rec_metric_comparison
A computer vision CNN DL project to create a face recognition model, comparing several types of loss metrics for training.

## Overview
In this project different loss metrics will be compared in order to train a model which will be able to perform a face recognition objective.
The objective will be tested by comparing the distances between output vectors from a model which is given face images as inputs. It will attempt to predict if 2 images are of the same person according to the distance between them.

The original loss metric which was used to do this project was a self created loss score based on the L2 distance between 2 vectors and the known fact if the images are of the same person or not (will be elaborated further below).
The other metrics compared are taken from (tf.contrib.losses.metric_learning):
1. contrastive loss
2. npairs loss
3. cluster loss

## Dataset
The dataset is a downloaded set of ~3500 face images divided into 700 classes.

Note: After finishing the project, it seems the choice of the dataset was not a good one since the amount of examples per class is very small.

## Model architecture
As the amount of data is very small, the architecture was built as a "thin" and "small" version of known models (like VGG16 etc.)
Different widths and depths were tested but this will not be elaborated in this project.
![alt text](https://github.com/michaelfiman/face_rec_metric_comparison/blob/master/face_rec_arch.PNG?raw=true)

## Loss metrics

### Self created loss
The main idea behind this loss is to penalize "similar" images if they are far from one another and "different" images if they are too close.
The model is first trained with a classification objective to give a good feature extraction at the one before last FC layer.
The the one before last FC layer is changed according to the following loss:
```
define: f(x) = output of one before last FC layer for input x
if label_image[a] == label_image[b]: (images are of same person)
  sim_loss += L2distance(f(a),f(b))
else: (images are of different people)
  diff_loss += max(0, threshold - L2distance(f(a),f(b))

loss = 0.5 * ((sim_loss/sim_count) + (diff_loss/diff_count))
```
Where threshold is a hyperparameter setting the margin required between different images, sim_count/diff_count are the amount of examples of each type.

Notes:
  1. Attempting to train the model only using this loss or updating all parameters (and not only the one before last FC) doesn't converge.
  2. In order to converge it is required to weight the different and similar losses to have the same affect.
  
Implemented in:
https://github.com/michaelfiman/face_rec_metric_comparison/blob/master/0_FERET_main_class_distance_loss.ipynb
  
### Contrastive loss
https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/contrastive_loss

The idea behind this loss is the same as the self implemented loss, except for using the distances as "energy" factors (squared distances) and no weighting between "diff" and "sim".
The metric is based on the followiing: https://arxiv.org/abs/1612.01213

Notes:
  1. As in the previos loss, it was required to run a classification objective and only train specific layers to get this loss to converge.
  2. low margin converged but gave bad results.

Implemented in:
https://github.com/michaelfiman/face_rec_metric_comparison/blob/master/1_FERET_main_contrastive_loss.ipynb


### npairs loss
https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/npairs_loss_multilabel

The idea behind this loss is to calculate "distances" using the inner product between a pair.
Using this "distance" calculation, a similarity matrix is created for each example in the input (where there will be 1 true and all the rest are false).
These are then used as logits for a softmax classifier problem. We can see that we expect 2 similar vectors to score a higher value when coputing an inner product.

This metric is based on the following: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

Notes:
  1. As in the previos loss, it was required to run a classification objective and only train specific layers to get this loss to converge.
  2. During training it seems that the model overfits very quickly and we need to take an early chckpoint for usage in the app.

Implemented in:
https://github.com/michaelfiman/face_rec_metric_comparison/blob/master/2_FERET_main_npairs_loss.ipynb

### cluster loss
https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/cluster_loss

The idea behind cluster loss is to try and get a greater "Oracle" cluster score using the ground truths as an understanding of the correct clustering, in comparison the clustering which is received by running a different clustring function.
The implementation is more complex and could be found in: https://arxiv.org/pdf/1612.01213.pdf

Notes:
  1. This model took much longer to train as some of the ops for this metric via tensorflow are not supported by the GPU in use.
  2. There was no need to pretrain this model on a classification objective, it was able to converge on the cluster objective alone.

Implemented in:
https://github.com/michaelfiman/face_rec_metric_comparison/blob/master/3_FERET_main_clustering_loss_metric.ipynb

## Results
Accuracy- the amount of correct predictions out of all predictions.
```
accuracy = correct_predictions/(correct_predictions+incorrect_predictions)
```
simm diff ratio- the ability to seperate between same and different pairs:
```
simm_diif_ratio = distance_average_sim/distance_avergage_diff
```
where distance_average_sim is the average distance computed for of a set of similar images
and distance_avergage_diff is the average distance computed for of a set of different images

![alt text](https://github.com/michaelfiman/face_rec_metric_comparison/blob/master/Result.PNG?raw=true)


## Summary and comparison
As expected, the first two metrics, self created and contrastive, scored more or less the same and had the same effect when running them in the app. This is due to the fact that they are very similar and train to reach the same general objective.

The npairs loss scored a better overall accuracy score with a much better sim to diff ration yet while running it in the app the reults were not so good for a one shot comparison (comparing one pair at a time). Although it does guess the correct most of the time it can sometimes guess an incorrect person or give a very low score for the correct person. I think that the inner product distance calculation can be very prone to changes in the input. Maybe training on images which would be collected from a similar device used on the app could have gave better results (video web cam VS still pictures).

The cluster loss scored pretty well but with a high sim diff ratio, and could have continued running to get a even higher score, yet when running in the app it was a total disaster. In think this is due to the fact that the clustering method was trying to cluster via ground truths of known faces and when receiving new and unknown faces, the clustering done in training has no meaning. I think this method could be maybe be better for differing between objects with less of a variance.

One more thing worth to notice is that the biggest problem was the choice off the dataset (3,500 images of 700 classes). I beleive that some of these metrics could have scored better if used with a dataset with more examples per class.

