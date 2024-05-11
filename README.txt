download the dataset and labels from the following links

https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ
https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=drive_link

then run the create_dataset.py script to create a dataset for the assignment, then run the code section in the colab file.


or simply clone the following repo or just run the corresponding cell in the colab file

https://github.com/Nikhil4902/CelebA_Dataset.git

this will download the dataset required for 50 classes i.e people. I chose to do this assignment for only 50 classes due to time constraints and availability of the GPU.

The pre trained model is already in the github repo so there is no need to train but you can do so. The training cells are marked so in the colab file.

I trained the model on Google's T4 GPU.

We can esaily train the model with a different number of classes in the create_dataset.py file, then run it and also change it in the colab file.
We might have to make a few changes to the top layers of our model to maintain the same level of accuracy even with more number of classes.



create_dataset.py
----------------------------
This file creates a dataset for training, testing and validating our model in a way that is compatible with keras's flow_from_directory function.
It first does face detection, crops and resizes the image to 150 * 150, resamples it and converts it to grayscale. This helps avoid backgrounds in most of the images and only keeps the subject.
We remove the color information as it is not that necessary for facial recognition.

face_recognition.ipynb
----------------------------
This file contains the main code for the model's training and testing.
I used a VGG16 pretrained model as the base and changed the top layers and trained on the dataset.
This model achieved an admirable accuracy of about 65%.
The model is already saved in the git repo so this can easily be verified by cloning the repo in the colab runtime and running the test cell.