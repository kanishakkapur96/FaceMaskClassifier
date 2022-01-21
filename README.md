# FaceMaskClassifier
README

This project includes following files,
1. aipersonfacemaskdetectorupdated.py :  In addition to part 1,this file performs operations listed below,
								A. Load data set used to detect bias
								B. Detect bias in the original model
								C. Loads new data to eliminate bias
								D. Training of model on new data
								E. Show training results of re-trained model
								F. Perform K-Fold cross validation on original model
								G. Perform K-Fold cross validation on re-trained model
								H. Displays the evaluation results
								

   Models are available on github. Github invite has been shared with the TA. The models are trained using GPU. To run the models ensure that the runtime environment has GPU enabled. To run on cpu, use map_location ='cpu' in torch load.

2. Dataset : We have used dataset mentioned below,
				A. Person with Mask: https://www.kaggle.com/omkargurav/face-mask-dataset
				B. Person without Mask: https://www.kaggle.com/omkargurav/face-mask-dataset, https://www.kaggle.com/cashutosh/gender-classification-dataset
				C. Not a person: https://www.kaggle.com/prasunroy/natural-images (excluding non person images)
	
	The data is divided into three parts,
		1. biasSet : Containes original test set, split into female and male categories. This is used to detect bias in the model. Relative path (finalDataset\biasSet\biasData\gender\)
		2. data: This directory contains original dataset used in submission 1. This is furthur split into train and test data
		3. newData : This directory contains updated dataset with additional training images.
	

PACKAGES REQUIRED: Following packages are required for running the python code,
					1. os
					2. torch
					3. torchvision
					4. pandas 
					5. sklearn
					6. seaborn
					7. matplotlib
					8. itertools

INSTRUCTIONS TO RUN THE CODE:
	1. USING GOOGLE COLAB/KAGGLE:
		A. For colab, place the attached data set to your google drive and mount the google drive in notebook. For kaggle, use zip dataset provided in submission and add it using Add Data feature on the notebook.
		B. Update the data_dir variable to your respective location.
		C. Add code in the notebook
		C. Run the code
	
	2. USING PYTHON CODE:
		A. Setup python 3.6 environment with above mentioned packages, preferrably with GPU
		B. Update the data_dir to point to location of dataset folder
		C. Run the code using command : python aipersonfacemaskdetectorupdated.py .py
		


