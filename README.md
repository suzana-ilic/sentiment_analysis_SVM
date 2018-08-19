# Sentiment Analysis

In Natural Language Processing, **Sentiment Analysis** refers to methods that systematically extract, classify and predict the polarity (positive or negative) of sentiment-bearing documents. Sentiment Analysis plays an important role in industry and is deployed in a wide range of application areas and domains, e.g. product reviews, customer service, marketing, recommender systems, online and social media monitoring, stock price predictions, healthcare applications, etc.

Sentiment Analysis can be done in rule-based settings using sentiment lexicons, or in the context of machine learning and deep learning, where systems learn from labeled data rather than rules or lexicons. In this case, we will implement a Support Vector Machine that learns from a dataset of ~100k positive and negative observations[1]. Performance depends on multiple factors, such as length of texts, noise, balanced or imbalanced classes, linguistic style, domain specific information, etc. We will work Twitter data with some characteristic traits, e.g. micro-length, colloquial, relatively noisy.

Related topics: Opinion mining, emotion recognition, sarcasm and irony detection.


# Support Vector Machines

**Support Vector Machines (SVMs)** are supervised machine learning models used for classification, regression and outlier detection.


The advantages of support vector machines are:

* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

In addition to performing linear classification, SVMs can efficiently perform non-linear classifications using the **kernel trick**, elevating and mapping the inputs into high-dimensional feature spaces.

There are many paramters, some of which are very sensitive. Some of the most important SVM parameters:

* **C :** float, optional (default=1.0). Penalty parameter C of the error term.
* **kernel :** string, optional (default=’rbf’). Specifies the kernel type to be used in the algorithm. 
    It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
* **gamma :** float, optional (default=’auto’). Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
* **class_weight :** {dict, ‘balanced’}, optional. Set the parameter C of class i to class_weight[i]*C for SVC. 
    If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
* **decision_function_shape :** ‘ovo’, ‘ovr’, default=’ovr’. Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
* **random_state :** int, RandomState instance or None, optional (default=None). The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.


## Here's a short intro to SVMs from Georgia Tech Machine Learning:

[![Support Vector Machine - Georgia Tech - Machine Learning](http://img.youtube.com/vi/eUfvyUEGMD8/0.jpg)](http://www.youtube.com/watch?v=eUfvyUEGMD8)


## References

[1] [Kaggle: SA Dataset](https://www.kaggle.com/c/twitter-sentiment-analysis2)

[2] [Pang and Lee (2008): Opinion mining and sentiment analysis.](http://www.cs.cornell.edu/home/llee/omsa/omsa.pdf)

[3] [Scikit-learn: SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

[4] [Scikit-learn: TF-IDF](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

[5] [Natural Language Toolkit (NLTK)](https://www.nltk.org/)

[6] [Keras: Pre-processing](https://faroit.github.io/keras-docs/1.1.0/preprocessing/text/)
