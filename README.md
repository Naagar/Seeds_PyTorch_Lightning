# Seeds_Classification


## Seeds_PyTorch_Lightning
## to Train your own model run main.py 



   Model-:        resnet18(pre_trained on Imagenet).
   
   Accuracy-:     train-84% , test-82%.
   
   No.of Epochs-: 250.

## Data_Set
   To download the dataset click on this link [DataSet](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sandeep_nagar_research_iiit_ac_in/EVXQD9ClwKtDvguuBsXefIgBexx27v2M8Ajhnwgl8-jixg?e=KklwXv)
which contains a zip file (data/train, data/test) 0.8 and 0.2 respectively  also the list of images names and their label(train_datafile.csv, test_datafile.csv) 
   Or  create a folder to download the images run Seed_Classification/seeds_dataset/download_images.py which will download images in the images folder.

   The dataset contains 4 classes: Discolored-0, Broken-2, Pure-1, Silkcut-3.
   Now we can run the main.py 
   
   Total images are 17802


## Models
   Each image is fed into the input layer, which then talks to the next layer, until eventually the “output” layer is reached. The network’s “answer” comes from this final output layer.
   Well, we train networks by simply showing them many examples of what we want them to learn, hoping they extract the essence of the matter at hand  and learn to ignore what doesn’t matter (like  shape, size, color or orientation).
   Loss and optimizer(Momentum and weight decay)

1. Resnet 18, 34, 50, 101, 152
2. MobileNetV2
3. Basic_model( 4 layers)
4. Squeezenet

   We explored large convolutional neural network models and trained for image classification, in a number ways.
   
   Choosing the model for classification is difficult as we have very small dataset and the distribution of the images of each class in the data set
And applied some image augmentation techniques.
   We have used the Resnet(18, 34, 50, 101, 152), SqueezeNet, MobileNetV2 and simple model (3 CNN and 1 FC layers).
   
   Here the main problem is the overfitting and to overcome this we have used the Drop out method. 



model(MobileNetV2)

   train accuracy is 98%.
   
   val_accuracy is 73%.
   
## model(resnet18)(updated)(e_200, optim-Adam,lr-0.001)

   train acc is 84%.
   
   test acc is  82%.
   
## Training Details



   Each RGB image was preprocessed by resizing the smallest dimension to 128, cropping the    center 128x128 region, subtracting the per-pixel mean (across all images) and then using 10 different sub-crops of size 128X128 (corners + center with(out) horizontal flips). 

   Stochastic gradient descent with a batch_size of 256 was used to update the parameters, starting with a learning rate of 10−3 , in conjunction with a momentum term of 0.1.

   We anneal the learning rate throughout training manually when the validation error plateaus.

## Future  Work

   We can do more experiments by changing the  Learning rate.
   
   Exploring the more image augmentation techniques.
   
   Visualizing the images after using image augmentation to know  more accurate effect of the same. 
   
   Decreasing the difference between the train and test accuracy.
   
   More experiments on the choosing the training model. 
   
   Improving overall accuracy. 
   
