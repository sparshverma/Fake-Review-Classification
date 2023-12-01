Fake Review 
Classification
Background: Consumers tend to rely heavily on reviews when making decisions about what to buy 
online. For a company like Amazon, which depends on this process, it is therefore particularly 
important that these reviews can be trusted. Your task will therefore be to develop a method for 
automatically classifying Amazon reviews as real or fake, to explore how plausible it is to automate 
this task. You will be working with a recently released corpus of Amazon reviews which have been 
manually analysed and annotated by the company itself (see https://s3.amazonaws.com/amazonreviews-pds/readme.html)
1
. Along with the review texts, which are labelled as either fake (_label1_) 
or real (_label2_), the data set contains a series of other features for each review (rating, verified 
purchase, product category, product ID, product title, review title). The corpus is made up of 21,000 
reviews, equally distributed across product categories, which have been identified as ‘non-compliant’ 
with respect to Amazon policies.
In this coursework, you will implement a Support Vector Machine classifier (or SVM) that classifies the 
reviews as real or fake. You will use both the review text and the additional features contained in the 
data set to build and train the classifier on part of the data set. You will then test the accuracy of your 
classifier on an unseen portion of the corpus. Much of the background for this part is in Unit 2 on Text 
Classification, though you should use all of your knowledge across the module.
Instructions: Follow the below instructions, and submit WELL DOCUMENTED code as one or more
IPython files (.ipynb) building on the template file NLP_Resit.ipynb as your starting point (Python 
3.7+). No separate report is required. You have the data in the file amazon_reviews.txt. Ensure your 
code runs from top to bottom without errors before submission. If you do use more than one 
IPython file, it must be clear which file corresponds to which questions.
The template file contains some functions to load in the dataset, but there are some missing parts 
that you are going to fill in as per the questions below.
1. (10 points) Start by implementing the parseReview and the preProcess functions. Given a line of a 
tab-separated text file, parseReview should return a triple containing the identifier of the review (as 
an integer), the review text itself, and the label (either ‘fake’ or ‘real’). The preProcess function should 
turn a review text (a string) into a list of tokens.
Hint: you can start by tokenising on white space; but you might want to think about some simple 
normalisation too.
2. (20 points) The next step is to implement the toFeatureVector function. Given a preprocessed 
review (that is, a list of tokens), it will return a Python dictionary that has as its keys the tokens, and
as values the weight of those tokens in the preprocessed reviews. The weight could be simply the 
number of occurrences of a token in the preprocessed review, or it could give more weight to specific 
words. While building up this feature vector, you may want to incrementally build up a global 
featureDict, which should be a list or dictionary that keeps track of all the tokens in the whole review 
dataset. While a global feature dictionary is not strictly required for this coursework, it will help you 
understand which features (and how many!) you are using to train your classifier and can help 
understand possible performance issues you encounter on the way.
1 See https://s3.amazonaws.com/amazon-reviews-pds/LICENSE.txt for the licensing information and terms and conditions 
for the use of the dataset.
Hint: start by using binary feature values; 1 if the feature is present, 0 if it’s not.
3. (20 points) Using the loadData function already present in the template file, you are now ready to 
process the review data from amazon_reviews.txt. In order to train a good classifier, finish the
implementation of the crossValidate function to do a 10-fold cross validation on the training data. 
Make use of the given functions trainClassifier and predictLabels to do the cross-validation. Make
sure that your program stores the (average) precision, recall, f1 score, and accuracy of your classifier 
in a variable cv_results.
Hint: the package sklearn.metrics contains many utilities for evaluation metrics - you could try 
precision, recall, fscore, support to start with.
4. (15 points) Now that you have the numbers for accuracy of your classifier, think of ways to improve 
this score. Things to consider:
• Improve the preprocessing. Which tokens might you want to throw out or preserve?
• What about punctuation? Do not forget normalisation and lemmatising - what aspects of this 
might be useful?
• Think about the features: what could you use other than unigram tokens from the review texts? 
It may be useful to look beyond single words to combinations of words or characters. Also the feature 
weighting scheme: what could you do other than using binary values?
• You could consider playing with the parameters of the SVM (cost parameter? per-class 
weighting?)
Report what methods you tried and what the effect was on the classifier performance in the notebook.
5. (15 points) Now look beyond textual features of the review. The data set contains a number of other 
features for each review (rating, verified purchase, product category, product ID, product title, review
title). How can the inclusion of these features improve your classifier’s performance? Pick three of 
these metadata types to use as additional features and report in the notebook how they improve the 
classifier performance
