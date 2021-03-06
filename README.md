# ai-liner: DV Hacks AI project repository

## Overview:
This is a supervised multiclass text classification algorithm that draws on tf-idf, random forest, support vector machines, naive bayes, and logistic regression in order to associate medical device keywords with FDA committee categories*.

This would be the first (post-processing) component of a data pipeline that would ultimately allow the end user - an investor performing due dilligence on a medical device startup - to input keywords related to the startup's medical device product, and receive a score indicating whether this medical device partnership is worth pursuing.

![Pitch Screenshot](https://github.com/nchitale/ai-liner/blob/master/pitch_screenshot.png)

## Key Technologies:
Scikit-learn for natural language processing and machine learning; Seaborn for visualization

## Steps to Build and Test:

We started with 2 datasets from the FDA: medical device product codes and premarket notifications. 
Combining the datasets based on product codes allowed us to associate Review Advisory Committees with medical device product category names. 

Here is how the number of medical devices released varies by advisory committee:
![Device Counts](https://github.com/nchitale/ai-liner/blob/master/device_counts.png)

Then, we used tf-idf to find the most correlated terms for each advisory committee.
The next step was model selection. We benchmarked four models: Random Forest Classifier, Linear Support Vector Machine, Multinomial Naive Bayes, and Logistic Regression. Out of these, the Linear Support Vector Machine model performed the best:

![Model Selection](https://github.com/nchitale/ai-liner/blob/master/model_selection.png)

Finally, we compared the accuracy between the predicted and actual committees for each medical device product name, visualized by the following confusion matrix:

![Confusion Matrix](https://github.com/nchitale/ai-liner/blob/master/confusion_matrix.png)

Using a chi-squared test we obtained the most correlated terms with each committee (for example, "oxygen" and "airway" for AN - Anesthesiology). 

*More info on FDA medical device committees available here: https://www.fda.gov/advisorycommittees/committeesmeetingmaterials/medicaldevices/medicaldevicesadvisorycommittee/default.htm

To view the model selection process, see "committees.py".
To get results from the selected model, download and run the "keywords-to-committees.ipynb" Jupyter notebook. Further instructions for downloading relevant data are within the notebook. Line 62 has a function that takes in a medical keyword and outputs the most associated FDA committee, and line 63 shows sample output for this function. Lines 64 and 65 publish the function to the TabPy server so that it can be linked with additional steps in the pipeline to score a medical device for viability.
