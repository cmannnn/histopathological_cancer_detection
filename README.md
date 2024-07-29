# Histopathological Cancer Detection

## Introduction

The Histopathological Cancer Detection aims to use cutting edge deep learning techniques to identify matastatic cancer in small images. The data supplied by Kaggle is a modified version of the PatchCamelyon dataset which contains more than 220,000 32x32 pixel images (in the .tif format) with a label of 0 for an example of a cell picture that does not contain a pixel of tumerous tissue whereas the 1 label is an example of a call p icture that does contain at least one pixel of tumerous tissue. The data is split into a train and test folder that contains the respective test and training images. Also included is a train_labels.csv file that provies the labels for the train dataset that can be used to train our CNN. 

For my implementation I will use a Convolutional Neural Network (CNN) using Pytorch to create an end to end model. The model will ouput the test id as well as a 0 or 1 label indicating the models prediction. 

## EDA

As part of the EDA process, I will start by visualizing the training_labels file. I will then build a basic histogram of the count of images that contains a cancer cell vs a count of the number of images that does not contain a cancer cell. I will then visualize 50 images from our train_labels file that contains a cancer cell and will separately visualize 50 images that does not contain a cancer cell. This will be the extent of the EDA performed as there is a limited number of columns to visualize in this dataset.

By doing this exploratory data analysis, both the creator and the viewer of the notebook will gain a better understanding of the structure and characteristics of the data that will be modeled later on in the notebook. 

## Data Augumentation

Data augumentation and normalization are critcal steps in deep learning. By flipping, rotating, translating, and zooming the pictures that the model will be trained on, the model learns additional features that it wouldn't have had exposure to if the augumentation did not happen. There is also an overfitting aspect to consider as well. If the effective size of the training data is increased due to the different variants of the image being passed to the model, the model is encouraged to learn general patterns in the data instead of the model learning specific examples. For these two reasons, data augumentation will be included in the deep learning model implementation. 

## Create DataLoader

Creating a DataLoader class facilitates data handling and managing in deep learning applications. The DataLoader class creates batches in the dataset that essentially groups multiple samples together which is more computationally efficient. If using a GPU, it also allows for parallel data loading which increases the speed at which the data is loaded. Overall, the DataLoader class is helpful in loading, preprocessing, and batching and can be incredibly helpful in managing large datasets such as the cancer dataset we are working with.

## CNN Model

The custom Convolutional Neural Network (CNN) was designed for the specific task of taking in the 32x32 images and outputting the binary cancerous or non-cancerous output. I decided to use a four layer technique with ReLu normalization and MaxPool pooling. The individual layers were then fully connecte that would be passed through the sigmoid function defined in the second code block below. The sigmoid activation function was particularly important because it is fit to output the binary response we were looking for. This was an obvious choice to use in the model.  

## Train Function

Below is a custom function that will be passed our model, the criteria, optimizer, scheduler and for a given epoch, will output the running model loss for both the training and validation set. It will also build a 'history' variable that will hold the training loss and accuracy after each epoch. The criterion, optimizer, and scheduler will be called in the cell below.

## Model Training

Here is where the model training is actually occuring. We will creat our CNN model, define our criterion loss to be binary cross entropy, set our optimizer to the Adam optimizer with a learning rate of 0.0001, and set the scheduler to help adjust the learning rate dynamically after every 7 epochs in our example. With the model, criterion, optimizer, and the scheduler defined, we can finally call the train_model function with our inputs from above to kick off testing.

In my case, the model took about 4.5 hours to train and achieved an overall accuracy score of .8987 on our validation set.

## Evaluation

In the evaluation phase, I observed steady improvement in both the accuracy and loss across both the training and validation loss and accuracy. In the earlier epochs, the model converged quicker whereas in the later epochs, the training accuracy and loss were only moderately effected. This slowdown in accuracy metrics indicates that our model is not overfit and is able to predict on both datasets quite well. The visualization of the training and validation loss/accuracy below graphically represents this slowdown. I found it very interesting to see that the accuracy in the validation set, the accuracy actually decreased from epoch 2 to 3. This indicates that a slightly lower learning rate could possibly achieve a higher overall accuracy.  

## Prdictions and Kaggle Test Submission

Finally, to submit our predictions to Kaggle we had to create a new test_dataset and test_loader using the same parameters from above. Then, a new variable 'predictions' was created to collect the models binary outputs. With the outputs, I had to conduct a bit of DataFrame manipulation to match the format in the sample_submission.csv and finally placed the final outputs into a 'submission.csv' file that will be uploaded to Kaggle. With my overall accuracy score of .8987 on the validation set, I'm hoping for a test accuracy in the mid 80%'s which would score quite well on the leaderboards. The first 5 rows of the final 'submission.csv' output can be visualized below.

## Conclusion

In conclusion, the task of detecting cancerous cells in histopathological images was no small task. It took very a very precise deep learning architecture to achieve the required outcomes. The process that was followed was to import the required packages, explore some of the training data, pre-process and augment the data using the the transformers class from sklearn and the DataLoaders class, building the model layer by layer, training the label with 15 epochs, and finally evaluating the models performance and preddiction generation which was submitted through the Kaggle competition homepage. Each step in the process was critical to achieve the overall accuracy of .8987.

Deep learning techniques such as CNN's are very promising in applications such as cancer detection that has saved numerous lives and will continue to do so as deep learning architecture and models become more and more robust. It was exciting (and also a bit of a struggle) to work through an entire end to end example throughout this notebook. There are certainly other applications for deep learning techniques such as this in NLP research, business optimization, and data generation to name a few.

I learned a great deal about how to conduct an end to end deep learning project and I hope others can achive the same outcome that I did.

### Challenges

There were many challenges that were overcome throughout the creation of this notebook. For example, handling the data imbalance in the initial data exploration phase. This was resolved by stratifying the data in our call to train_test_split by using the stratify method that allowed for the same number of cancerous and non-cancerous samples to be present in both the training dataframe and the validation dataframe. This certainly helped improve our models ability to detect both classes.

Another challenge throughout this notebook was getting the CNN architecture correct. At first, I was very lost in what was happening in the different layers and how they were able to connect. After referencing some additional sample notebooks, I was able to figure it out but it certainly wasn't clear to me at first. Since this was the first real end to end deep learning model I have created, the intricate steps required to even achieve an output was hard for me. It was helpful to reference additional online material and read the PyTorch docs.

### Future Work

In the future, the following points could be addressed to achieve an even more impressive accuracy:
- Hyperparameter tuning: although some hyperparameter tuning was attempted, further tuning hyperparameters such as learning rate and batch size could result in a higher overall accuracy.
- Transfer Learning: Implementing transfer learning by leveraging pre-trained models could grealty enhance my models performance. I noticed many examples of transfer learning in the shared notebook in the Kaggle competitions homepage that received almost perfect results.
- Ensemble methods: Using ensemble methods to combine predictions from multiple models could also result in a higher overall accuracy. For example, creating a custom model paired with a pre-trained model would be very interesting architecture to research further.
