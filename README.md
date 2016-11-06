Doppler
=====
####Adapting Well-Trained Deep Learning Models to iOS Devices for Recognition Tasks. (Chao-Ming Yen)

##Summary
The project aims to apply recent model compression techniques (e.g. quantization, XOR-Net) on a pre-trained deep learning model, and optimizes efficiency of the compressed model on iOS devices for recognition tasks.

##Background
While deep learning seems to be taken for granted nowdays, application of deep learning on mobile devices is still challenging. A deep learning model can be well trained off-line on a super GPU cluster. But when trying to use the trained model on mobile devices for inference tasks, it brings serious problems. First, the RAM on mobile devices is limited. A deep learning model requires too much run-time memory for a mobile device to perform a single forward propagation. Second, the inference speed of a model and energy issue also matters. Some tricks such as vector quantization and XOR-Net are designed to preprocess pre-trained models before deploy them on mobile devices. In sum, the Doppler project aims to find the best way to compress a well-trained model so that it can be applied to mobile devices.

##The Challenge
The challenge lies in making a pre-trained model more efficient while keeping(or even improving) its original accuracy. While some conventional compression techniques such as network pruning and vector quantization be applied, few could tell the consequences after the compression. This is because "how deep learning works" is still a mystery, and there are many unknown details waiting to be found. Since current researches about deep learning are mostly focusing on improving the accuracy instead of the efficiency of a model, finding the best way to compress a model could serve as challenging task here. Nonetheless, deep learning serves as powerful feature extractor and has great potential in mobile computer vision tasks. It's still worth the effort deploying these models on mobile devices.


##Goals & Deliverables
Describe the deliverables or goals of your project.
    - In a couple of sentences separate your goals into what you PLAN TO ACHIEVE (what you believe you must get done to have a successful project and get the grade you expect) and an extra goal or two that you HOPE TO ACHIEVE if the project goes really well and you get ahead of schedule.
    - Describe what success looks like and how it can be evaluated. For example, if your project is to measure the velocity of a baseball being thrown in front of an iOS device, how will you validate that it works? Screen shots of the App in action? A speed benchmark run across a variety of videos? A live video of the app in action? It will NOT be enough to simply provide the Xcode project - you will need to provide evidence that you have achieved your goal.
    - How realistic is it for your team to get what it needs to get done within the allotted time? Remember you only have a few weeks to get this project completed.

##Schedule
 - by Nov. 11, 2016: Setup the Torch-iOS environment. Study XOR-Net
 - by Nov. 18, 2016: Deploy a working deep learning model on iPad 2.
 - by Nov. 25, 2016: Implement XOR-Net version of the model on iPad 2.
 - by Dec. 02, 2016: Fine-tune the model with compression techniques.
 - by Dec. 09, 2016: Prepare for the final presentation and report.
 