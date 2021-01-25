# MU-Net
U-Net Model for Sclera Segmentation in the Mobile Environment using a Transfer Learning approach on MobileNetV2


This repository contains the implementation of a U-Net architecture using Keras with Tensorflow at its backened for segmenting Sclera using a Transfer Learning approach. 
 ## 1- Proposed method

The proposed approach employs a U-Net inspired model conditioned on MobileNetV2 class features to segment sclera and background in an eye, where two-stage fine-tuning was applied to the MobileNetV2 model. The data was augmented by different models.

In our method, we used U-Net [U-Net] with Pretrained MobileNetV2 [MobileNetV2]. The U-Net is based on the fully convolutional network and we modified its architecture to work with fewer training samples and to achieve more accurate segmentations. We used the pretrained weights provided with MobileNetV2 for the ImageNet dataset [ImageNet] and fine-tuned it on the sclera domain. To provide domain adaptation, we fine-tuned the MobileNetV2 model on the provided data for SSBC 2020 from the M(A)SD. We also augmented the data by performing left-right flips. MobileNetV2 has less parameters, due to which it is easy to train. For the binary masks, we chose the binarization threshold by iterating over the images in the provided dataset and applied the threshold that achieved the highest F1-score.

We integrate UNet deep neural network and pre-trained MobileNetV2. The MobileNetV2 was trained on the ImageNet. 
MobileNetV2 is easy to train because it has less parameters.
A pre-trained MobileNetV2 alone with UNet model in comparison with a Unet model achieve high performance.

#    2. RGB or grey-scale images?

RGB images.

  #  3. data preprocessing or data augmentation? 
We performed data augmentation and created 2 images per original image in the SSBC training dataset. For augmentation, we generated the flipped version of the image and used both in our model training. We also resized the images to 224x224 size.

 #   4. What loss function or learning objectives did you use? 
We used a Dice coefficient to measure overlap of between two images as loss function. Dice coefficient term is standard in segmentation tasks and it is ranges from 0 to 1. Zero indicate worse overlap in contract one indicate perfect overlap.

  #  5. How many parameters had to be learned during the training stage?
Modified deep neural network requires training and consists of 416,209 total parameters and as result 409,025 trainable parameters.

  #  6. Hardware for the experiments (CPU, GPU, RAM, other relevant info)? 
    
CPU: AMD Ryzen Threadripper 1920x
RAM: Corsair CMW64GX4M4C3200C16 RGB Pro 64GB (4 x 16GB) DDR4 3200 MHz C16 
GPU: NVIDIA Quadro P5000
We used the GPU for CNN model training.

 #   7. Programming language? 
    
For model training, the binarization threshold computation and thresholding process we used the Keras and Tensorflow framework and Python.
Prerequisites

# The following dependencies are required to run the U-Net model:

Keras with Tensorflow at it's backend
numpy
skimage
Pillow
OpenCV
h5py
Scikit learn
Matplotlib

# 8. Running the tests
For running the model for training and testing, make sure all dependencies are installed properly.

Then run the following,
# [optional]
if you want to augment your data you can run first the following:
python Augment_datasets.py

This will will create the augmented images and masks for all the image sets which can be used for training

## Now run the following,

python MobV2-U-NetSSBC2020.py
This will save the model weights in current folder.

# Testing
Run the following

python predict_test_IMG.py
All the results will be saved in the folder with the mentioned name.
