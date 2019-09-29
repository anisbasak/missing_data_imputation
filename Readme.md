Requirements
Python 3 (Version 3.5 or below)

Install Libraries
1. tensorflow
2. tensorflow-gpu
3. keras
4. scikit-learn
5. pandas
6. matplotlib
7. keras-gpu
8. numpy 


Files:

1. evaluator_function.py
This file contains the evaluator function in python

2. train_initial_baseline_solutions.py
This file contains the initial basic imputation techniques like mean, median, rolling window

3. train_many_to_many.py
This file contains the LSTM network to train the model

4. imputation.py
This file imputes the values for the missing data in the patients.
To run this code we to use a specific folder structure.
	- make a folder 'test_data' at the same level as imputation.py
	- paste the naidx.csv in the 'test_data' folder
	- make a folder within 'test_data' as 'test_with_missing'
	- paste all the files(csv) of patients within the 'test_with_missing' folder
	- when the file is run a folder 'train_data/processed_many_to_many' will be created at the level of imputation.py
	- all the imputed files for the patients will be present in the 'train_data/processed_many_to_many' folder


Folder

1. model
This folder contains the models saved after training for 6000 patients.
We load this model to get the imputed values.

