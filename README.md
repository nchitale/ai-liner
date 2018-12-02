# ai-liner
DV Hacks AI project repository

We used 2 datasets from the FDA: medical device product codes and premarket notifications. 
Combining the datasets based on product codes allowed us to associate Review Advisory Committees with medical device product category names. 

Here is how the number of medical devices released varies by advisory committee:
![Device Counts](https://github.com/nchitale/ai-liner/blob/master/device_counts.png)

Then, we used tf-idf to find the most correlated terms for each advisory committee.
The next step was model selection. We benchmarked four models: Random Forest Classifier, Linear Support Vector Machine, Multinomial Naive Bayes, and Logistic Regression. Out of these, the Linear Support Vector Machine model performed the best:

![Model Selection](https://github.com/nchitale/ai-liner/blob/master/model_selection.png)
