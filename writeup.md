# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
1. Load the data set (see below for links to the project data set)
2. Explore, summarize and visualize the data set
3. Design, train and test a model architecture
4. Use the model to make predictions on new images
5. Analyze the softmax probabilities of the new images
6. Summarize the results with a written report

[//]: # (Image References)

[randomTrainingDataSet]: writeup/randomTrainingDataSet.png "Random Training Input Images"
[originalSpreadOfInput]: writeup/originalSpreadOfInput.png "Input data bar chart"
[shiftedImage]: writeup/shiftedImage.png "After image shift"
[rotatedImage]: writeup/rotatedImage.jpg "After image rotation"
[histogramEqualization]: writeup/histogramEqualization.png "After histogram Equalization"
[image1]: own-images/1.png "Web Image 1"
[image2]: own-images/2.png "Web Image 2"
[image3]: own-images/3.png "Web Image 3"
[image4]: own-images/4.png "Web Image 4"
[image5]: own-images/5.png "Web Image 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

* [environment.yml](https://github.com/anvaysrivastava/CarND-Traffic-Sign-Classifier-Project/blob/master/environment.yml) : Contains added dependency in order to run the notebook
* [Traffic_Sign_Classifier.html](http://anvay.xyz/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html) : HTML view of one specific run
* [Traffic_Sign_Classifier.ipynb](https://github.com/anvaysrivastava/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) : The notebook itself
* [own-images/*.png](https://github.com/anvaysrivastava/CarND-Traffic-Sign-Classifier-Project/tree/master/own-images) : Images downloaded from web as extra data samples.
* [writeup.md](https://github.com/anvaysrivastava/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md) : Writeup for the project. Copied it to [Wiki for better view](https://github.com/anvaysrivastava/CarND-Traffic-Sign-Classifier-Project/wiki/Writeup)

### Dataset Exploration

#### [Dataset Summary](http://anvay.xyz/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html#Basic-Summary-of-the-Data-Set)

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### [Exploratory Visualization](http://anvay.xyz/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html#Visualization-of-the-images-and-labels)

Random images provided in training set are
 ![alt text][randomTrainingDataSet]

Here is an exploratory visualization of the data set. It is a bar chart showing how the classes are distributed

![alt text][originalSpreadOfInput]

### Design and Test a Model Architecture

#### [Preprocessing](http://anvay.xyz/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html#Pre-process-the-Data-Set)
3 types of preprocessing are done before training on data set
1. **Augment the data**
* This is done because as you can see from exploratory visualization that some classes have very low frequency. Hence I rotate and shift the images of classes that occured less.
Shited Image ![alt text][shiftedImage]
Rotated Image ![alt text][rotatedImage]
2. **Convert to grascale and then perform histogram equalization**
* This is done because the brightness varies a lot on random training images. histogram equalization ensures that the data does not fluctuate on brightness.
![alt text][histogramEqualization] 
3. **Normalize the input**
This is done since having variables with 0 mean makes the network faster.

#### [Model Architecture](http://anvay.xyz/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html#Model-Architecture)

I started with lenet, which was saturating at 0.80 - 0.82. 

Post which I made the network relatively wider to accomodate for more classes. This worked really well for me.

Making the network any wider was taking more than 30 epoch to peak the training accuracy. Making the network any thinner was reducing the max training accuracy.

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 Normalized, Histogram equalized image                             | 
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x12    |
| RELU                  |                                               |
| Pool                  | 2x2 stride, window 2x2, outputs 14x14x12  |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x32    |
| RELU                  |                                               |
| Pool                  | 2x2 stride, window 2x2, outputs 5x5x32  |
| Flatten           | output 800                |
| Fully connected       | input 800, output 120 |
| Fully connected       | input 120, output 84 |
| Fully connected       | input 84, output 43 |

**My loss function was same as lenet which was `loss_function = reduce_mean(entropy)`**
Where `entropy = cross entropy(softmax(model output), one_hot_y)`


#### Model Training

The hyper parameter tuned were
1. Epoch count: The validation accuracy was not increasing after 15 epochs. Hence I chose that value to avoid overfitting.
2. Batch Size: In every epoch I wanted 1 image of each class to be present. Hence a random batch size of 50 was chosen give the elements in each batch were randon. Having batch size more than this was leading to much lesser accuracy in early epochs as well as slower rate of convergence.

#### Solution Approach

I approached the problem with the following steps.
1. Test lenet model. I was getting a training accuracy of 0.85 - 0.9 on repeated runs with batch size 200.
2. On analyzing the input, I realized that the brightness of images varied a lot, as well as the distribution of classes was not even. Hence I did the preprocessing as mentioned earlier.
3. The above steps did not change the training accuracy by much lot, but the rate of convergence got much better and I was reading 0.8 accuracy in 4-5 epochs.
4. At this point I widened the entire model and training accuracy shot up to 0.95-0.97

My final model results were:
* training set accuracy of 0.973
* validation set accuracy of 0.921 
* test set accuracy of 0.901


###Test a Model on New Images

#### Acquiring New Images

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign             | Stop sign                                     | 
| U-turn                | U-turn                                        |
| Yield                 | Yield                                         |
| 100 km/h              | Bumpy Road                                    |
| Slippery Road         | Slippery Road                                 |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .60                   | Stop sign                                     | 
| .20                   | U-turn                                        |
| .05                   | Yield                                         |
| .04                   | Bumpy Road                                    |
| .01                   | Slippery Road                                 |


For the second image ... 