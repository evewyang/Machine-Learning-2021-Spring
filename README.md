# Machine-Learning-2021-Spring
This folder concludes my work suring the Honors Machine Learning class at NYU.<br><br>
<strong>Subject Matter</strong>: review of probability and statistics, K-Means clustering, KNN regression and classification, curse of dimensionality, Optimization and Gradient Descent, Linear Regression, Poly Regression, Bias/Variance, Regularization, Classification/Logistic Regression, Probabilistic generative model, SVM, Trees, Bagging and Boosting, deep learning, Neural Network, activation functions, CNN, RNN, PCA, autoencoder<br><br>

<strong>[Homework 1](./homework_1)</strong>: Probability and Statistics (deriviation of conditional probability, independency); KNN regression and classification (analysis of overfitting, hyperparameter learning, train-test error); K-Means clustering (elbow plot, cluster-labeled sactter plot)<br><br>

<strong>[Homework 2](./homework_2)</strong>: Linear Regression (analysis of unbiased estimator of coefficient, gradient descent & learning rate); Ridge Regression (empirical risk minimization); Feature Correlation (multicollinearity, Lasso objective function); Pickle (model serialization)<br><br>

<strong>[Homework 3](./homework_3)</strong>: Probabilistic linear classifiers (discriminative linear classifier: logistic regression; generative linear classifier: Linear Discriminant Analysis/LDA); Naive Bayes (spam email identification: pre-processing text data, bag-of-words, maximum likelihood estimator) <br><br>
<strong>[Final Project](./Final_Project)</strong>: Emotion Recognition with Deep Learning -- Modeling and Structural Analysis
<ul>
<li><strong>Dataset</strong>: Format: (picture, label), Picture: 350x350 pixel. Mostly Grayscale, a few RGB with different dimension; Label: {'anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'}</li>
<li><strong>Preprocessing</strong>: Resizing: reshape the pictures into 32x32 pixel; Channel Matching: transfer pictures into 3-channel RGB; Input Format Aligning: 13690x3072 numpy array of uint8s (machine learning) or 4D numpy array of dimension (sample_size, 32,32,3) (deep learning)</li>
<li><strong>Model Training</strong>: Logistic Regression, KNN, Decision Tree/Random Forest, CNN with different architectures(VGG (VGG11, VGG13, VGG19), GoogLeNet, SEResNet (SEResNet18, SEResNet34), InceptionV3, MobileNet) with stratified bootstrapping</li>
<li><strong>Network Architecture Break-Down</strong>: layers(organization, complexity, depth & width), classifiers, weight, regularization</li>
<li><strong>Testing Accuracy Comparison</strong>: Best 75.8% (SEResNet18), overall 60%+</li>
<li><strong>Future Works</strong>: better distinguishing data labels (first identify sentiments, then detailing emotions within such sentiment)</li>
</ul>

Packages and Libraries Covered: NumPy, Pandas, Matplotlib, SKlearn, PyTorch<br><br>
Relevant Material: [1] James G. et al. In Introduction to Statistical Learning; [2] Hastie T. et al. The Elements of Statistical Learning.
