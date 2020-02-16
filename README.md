![Titanic harbor](https://github.com/bluejurand/Titanic-python/blob/master/images/Titanic_harbor.jpg)  
# Titanic-python
![Build status](https://travis-ci.org/bluejurand/Titanic-python.svg?branch=master) 
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg) 
![Python 3.6](https://img.shields.io/badge/python-3.3-blue.svg) 
![Jupyter Notebook 5.4.0](https://img.shields.io/badge/jupyter_notebook-5.4.0-orange.svg) 
![Scikit-learn 0.19.1](https://img.shields.io/badge/scikit_learn-0.19.1-orange.svg) 
![Pandas 0.22.0](https://img.shields.io/badge/pandas-0.22.0-green.svg) 
![Numpy 1.12.1](https://img.shields.io/badge/numpy-1.12.1-yellow.svg) 
![Scipy 1.0.0](https://img.shields.io/badge/scipy-1.0.0-blue.svg) 
![Matplotlib 2.1.2](https://img.shields.io/badge/matplotlib-2.1.2-blue.svg) 
![Seaborn 0.8.1](https://img.shields.io/badge/seaborn-0.8.1-black.svg)  
Repository for Titanic-kaggle dataset, with data analysis and testing different classification algorithms.
Link to original kaggle competition: https://www.kaggle.com/c/titanic

## Motivation

To practice data analysis, feature engineering, missing value imputation and different classification algorithms in python for Titanic passenger dataset.

## Installation

Python is a requirement (Python 3.3 or greater, or Python 2.7). Recommended enviroment is Anaconda distribution to install Python and Jupyter (https://www.anaconda.com/download/).

__Installing dependencies__  
To install can be used pip command in command line.  
  
	pip install -r requirements.txt

__Installing python libraries__  
Exemplary commands to install python libraries:  
 
	pip install numpy  
	pip install pandas  
	pip install xgboost  
	pip install seaborn 
	
## Code examples

	#Define a function which will replace the NaNs for age with mean, for each passenger class.
	def impute_age(cols):
		Age = cols[0]
		Pclass = cols[1]
		
		if pd.isnull(Age):

			if Pclass == 1:
				return 37

			elif Pclass == 2:
				return 29

			else:
				return 24

		else:
			return Age  
<!-- -->
	#Use grid search and cross-validation to tune the model
	predictors = [x for x in train_scaled.columns]
	param_test1 = {
	 'max_depth':range(3,12,2),
	 'min_child_weight':range(1,8,2)}
	gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, n_estimators = 140, max_depth = 5,
	 min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8,
	 objective = 'binary:logistic', nthread = 4, scale_pos_weight = 1, seed = 27, return_train_score = False), 
	 param_grid = param_test1, scoring = 'roc_auc', n_jobs = 4, iid = False, cv = 5)
	 gsearch1.fit(train_scaled[predictors], y_train)
	 gsearch1.cv_results_['params'], gsearch1.best_params_, gsearch1.best_score_

## Key Concepts
__Machine Learning__  

__Classification__  

__Cross-Validation__  
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html

__Model Evaluation__  
  -Scores  
  -Classification reports  
  -Visualization tools  
  -Precision recall

__XGBoost__  
https://xgboost.readthedocs.io/en/latest/  
  
![Features importance](https://github.com/bluejurand/Titanic-python/blob/master/images/Features%20importance.png)