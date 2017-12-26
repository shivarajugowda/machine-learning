# Passenger Screening Model to Imporve TSA's Threat Recognition in Airports.
## Capstone Proposal
Shivaraju Gowda  
December 17th, 2017

### Domain Background
Airport security is critical and necessary part of a safe travel. As anyone who has travelled on an airplane knows, this comes at the cost of long lines and wait times at the airport security checkpoint. The U.S. Transportation Security Administration (TSA) is responsible for all U.S. airport security, they screen more than two million passengers daily. They are very aware of their seemingly conflicting goals of thorough security screenings and short wait times to keep their customer safe and happy at the same time.

TSA has identified high false alarm rates as creating significant bottlenecks at the airport checkpoints. Whenever sensors predict a potential threat, staff needs to engage in a secondary, manual screening process that slows everything down. As the number of travelers increase every year and new threats develop, the prediction algorithms need to continually improve to meet the increased demand.

Currently, TSA purchases updated algorithms exclusively from the manufacturers of the scanning equipment. These algorithms are proprietary, expensive, and often released in long cycles. TSA is stepping outside their established procurement process and is challenging the broader data science community to help improve the accuracy of their threat prediction algorithms as a part of the Kaggle competition. Using a dataset of images collected on the latest generation of scanners, participants are challenged to identify the presence of simulated threats under a variety of object types, clothing types, and body types. Even a modest decrease in false alarms will help TSA significantly improve the passenger experience while maintaining high levels of security.

Kaggle Competition Website: https://www.kaggle.com/c/passenger-screening-algorithm-challenge

### Problem Statement
The input contains a large number of body scans of subjects at different granularity levels acquired by a new generation of millimeter wave scanner called the High Definition-Advanced Imaging Technology (HD-AIT) system. The task is to predict the probability that a given body zone (out of 17 total body zones) of the subject has a threat present. There can be multiple threats or no threats in a subject.


### Datasets and Inputs
The images in the dataset are designed to capture real scanning conditions. They are comprised of volunteers wearing different clothing types (from light summer clothes to heavy winter clothes), different body mass indices, different genders, different numbers of threats, and different types of threats. Due to restrictions on revealing the types of threats for which the TSA screens, the threats in the competition images are "inert" objects with varying material properties. These materials were carefully chosen to simulate real threats.

The volunteers used in the first and second stage of the competition will be different (i.e. your algorithm should generalize to unseen people). In addition, you should not make assumptions about the number, distribution, or location of threats in the second stage.

The data for each scan performed by the HD-AIT system is referred to as an HD-AIT Frame. A frame consists of the following four binary files:

_.ahi = calibrated object raw data file (2.26GB per file)
_.aps = projected image angle sequence file (10.3MB per file)
_.a3d = combined image 3D file (330MB per file)
_.a3daps = combined image angle sequence file (41.2MB per file)

### Evaluation Metrics
Log loss is used as an evaluaiton metric for this competition. If there are N images, our model will be making 17N predictions. Submissions are scored on the log loss:

−1/N∑i=1N[yilog(ŷ i)+(1−yi)log(1−ŷ i)],

where:

N is the 17 * the number of scans in the test set
ŷ is the predicted probability of the scan having a threat in the given body zone
yi is 1 if a threat is present, 0 otherwise
log() is the natural (base e) logarithm

Note: the actual submitted predicted probabilities are replaced with max(min(p,1−10−15),10−15). A smaller log loss is better.

### Solution Statement
_(approx. 1 paragraph)_

In order to solve this problem. I am going to use this two pieces of information. 
1) Object detection improvments using maching learning. 
2) The invariability of human body on the vertical axis. 

The first will help us detect a contrband/threat in an image. The second will help in figuring out the body zone the threat belongs to. 

Object detection has improved quite a bit in the last couple of years. There are now prepackaged softward which I will benchmark and leverage to handle this step. Specifically I will be using TensorFlow Object Detection API(https://github.com/tensorflow/models/tree/master/research/object_detection). The threats come in different size and shapes. I was initially skeptical if treating all the threats as the same label would work, but experiments showed that it was sufficient. 

I did some research as to how the human body varies in size and proportion. I found that the human body doesn't vary that much verically as compared to horizontally. The sizes are proportional too. Say for example the lenght of the body could be talked in terms of the lenght of the head. I  used this information to detect head, groin and hands and based on the position of these body part and the frame number I could predict the zone of a contraband with more than 98% accuracy. I will use this information to build a model to generate labels for a secondary model to predict the body zone a contraband belongs to. 

So here is the flow of my model. 

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

#### Object Detection:
Use Tensorflow Object Detection API.
1) Benchmark different 
2) Use Google Cloud.
3) Improve accuracy. 

#### Zone Prediction:
1) Custom predictor. 
2) XGBoost. 

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
