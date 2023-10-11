# Using Python CART for Detecting the Most Predictive Factors of Health Index


## Motivation

 The goal of my project is to apply decision tree analysis to a county-level dataset (project_477.csv) to detect the most predictive factors of the health index. 
 The dictionary of the dataset can be found in [FP_data_dictionary.doc.pdf](https://github.com/wangshasha111/US_county_poor_health_prediction/blob/main/FP_data_dictionary.doc.pdf)
 
## Data Source
 The data concerns health, economic and demographic characteristics of counties in the US. The unit of analysis is the county. If you are interested in the data source it is [https://www.countyhealthrankings.org/](https://www.countyhealthrankings.org/). 

## Methodolody
 To detect the most predictive factors of the outcome variable - health index, I employ CART (Classification And Regression Trees), the method of decision tree analysis with cost complexity pruning.
 CART is a nonparametric modeling approach used in statistics, data mining, and machine learning for classification and prediction.
 It is considered a model-free approach. I use the ``sklearn" package of Python 3.8.3 to estimate and plot the CART trees. 

## Software
I use Python 3.8.3 with packages **pandas seaborn numpy matplotlib sklearn math patsy**.

## Takeaway
 The takeaway from the project is that the most predictive factors of a county's health index are, in order of significance, life expectancy, insufficient sleep, smoking and excessive drinking, adult obesity, and the percentage of females.


## Outputs
 I output a jupyter notebook FP_starter_ShashaWang.ipynb containing both the codes and the description of each step of the project. Its PDF and html version are also provided in [FP_starter_ShashaWang.pdf](https://github.com/wangshasha111/US_county_poor_health_prediction/blob/main/FP_data_dictionary.doc.pdf) and [FP_starter_ShashaWang.slides.html](https://github.com/wangshasha111/US_county_poor_health_prediction/blob/main/FP_starter_ShashaWang.slides.html)






 








