Doppler
=====
###Compressing Deep Learning Models for Artistic Style Transformation on iOS Devices. (Chao-Ming Yen)
(https://www.youtube.com/watch?v=9KOaqMKK1mk)

##Summary
The project aims to apply recent model compression techniques (e.g. quantization, XOR-Net) on a pre-trained deep learning model, and optimizes efficiency of the compressed model on iOS devices for artistic style transformation tasks.


##Background
While deep learning seems to be taken for granted nowdays, application of deep learning on mobile devices is still challenging. A deep learning model can be well trained off-line on a super GPU cluster. But when trying to use the trained model on mobile devices for inference tasks, it brings serious problems. 

First, the RAM on mobile devices is limited. A deep learning model requires too much run-time memory for a mobile device to perform a single forward propagation. Second, the inference speed of a model and energy issue also matters. Some tricks such as vector quantization and XOR-Net are designed to preprocess pre-trained models before deploy them on mobile devices. Third, the size of a deep learning model is usually gigantic for a mobile application, and it becomes a problem when designing a computer vision app < 100 mb. In sum, the Doppler project aims to find the best way to compress a well-trained model so that it can be applied to mobile devices.


##The Challenge
The challenge lies in making a pre-trained model more efficient while keeping(or even improving) its original accuracy. While some conventional compression techniques such as network pruning and vector quantization be applied, few could tell the consequences after the compression. This is because "how deep learning works" is still a mystery, and there are many unknown details waiting to be found. Since current researches about deep learning are mostly focusing on improving the accuracy instead of the efficiency of a model, finding the best way to compress a model could serve as challenging task here. Nonetheless, deep learning is no doubt a powerful feature extractor and it has great potential in mobile computer vision tasks. It's still worth the effort deploying these models for mobile applications.


##Goals & Deliverables
The minimum goal of Doppler project is to compress a pre-trained deep learning model (e.g. AlexNet, VGG, GoogLeNet) and deploy them on an iOS device (iPad 2, specifically) to serve as a feature extractor and use it to either perform classification or recognition tasks. If time permits, a better form of end-use application such as pixel-wise object segmentation can be considered in this project as extra effort.

To evaluate the project, the results of different model compression techniques will be benchmarked, in the aspect of model size, inference speed and inference accuracy. A successfully compressed model will have less parameters, smaller size in space, faster inference speed, and equal accuracy as the original pre-trained model. Some charts will be created to show the comparison across the performances. In addition, a live video of the final application will be recorded to show how the application can benefit from the increase of inference speed. Therefore, the final deliverables of this project will be:

 - Source code of Xcode project
 - A demo video
 - A report that discusses and summerize the benchmarking results

Since the allotted time is limited in few weeks, reusing existing codes is crucial. Therefore, the first step of this project is to use public resources that can currently be found to build a prototype in a short time. For example, many published deep learning models have their own dedicated GitHub repository, and the source code can serve as a cornerstone for this project.


##Schedule
 - by Nov. 11, 2016: Setup the Torch-iOS environment. Study XOR-Net
 - by Nov. 17, 2016: Deploy a working deep learning model on iPad 2. (checkpoint)
 - by Nov. 25, 2016: Implement XOR-Net version of the model on iPad 2.
 - by Dec. 02, 2016: Fine-tune the model with compression techniques.
 - by Dec. 09, 2016: Prepare for the final presentation and report.
 
