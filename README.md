# Naive-Bayes-Document-Classification
Implementation of several text classification systems using the naive Bayes algorithm and semi-supervised Learning 


| Dataset  | http://qwone.com/~jason/20Newsgroups/ |
| ------------- | ------------- |
| BinaryMultinomialNaiveBayes.py  | Implementation of Multiclass Naive Bayes with Binary Bag of Words **(B-BoW)** representation  |
| MultinomialNaiveBayes.py | Implementation of Multiclass Naive Bayes with Count Bag of Words **(C-BoW)** representation  |
| TfIdfNaiveBayes.py | Implementation of Multiclass Naive Bayes with  **tf-IDF** representation | 
| EM Supervised Learning (BinaryMultinomialNaiveBayes.py) | I used semi supervised learning to do document classification. The underlying algorithm I use is Naive Bayes with the B-BOW representation | 
| Filtering Technique for EM  | top-k: Filter the top-k instances and augment them to the labeled data | 
| Filtering Technique for EM | Threshold: Set  a  threshold.   Augment  those  instances  to  the  labeled  data  Thewithconfidence higher than this threshold| 

The training and testing accuracies and comparison of various representations can be found in the report. 
