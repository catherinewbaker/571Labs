import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape # 1600, 13989
        
        # classes = a list of possible classes
        classes = np.unique(y) # [ 0 1 ]
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0] # 2
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes) # prob of a doc being a certain class without looking into doc components
        likelihood = np.zeros((n_words,n_classes)) # prob of observing a word given that the doc is a certain class

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being 
            # active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 
            # corresponds to the fifth feature!
        
        ###########################

        # Step 1: Define priors as the probabilities of a given document to be negative (index 0) or positive (index 1)
        #          A. sum the total positive and negative review count using y
        for document in y:   
            prior[document] += 1
        #          B. Divide the sums by the total number of reviews
        for index, clss in enumerate(prior):
            prior[index] /= n_docs

        # Step 2: Define likelihood probabilities of a given word to be positive or negative
        #          A. pull the frequency of each word per document and add it to a tally in likelihood, organized by class
        for (xVal, yVal), wordFreq in np.ndenumerate(x):
            likelihood[yVal][y[xVal]] += wordFreq
         #         B. Find the total word frequency per class by adding the columns of likelihood
        negNum = 0
        posNum = 0
        for wordIndex in range(n_words):
            negNum += likelihood[wordIndex][0]
            posNum += likelihood[wordIndex][1]
        #          C. Now perform Add-1 Smoothing
        for (xVal, yVal), wordFreq in np.ndenumerate(likelihood):
            likelihood[xVal][yVal] += 1
        #          D. Divide the current word frequencies in likelihood by the total frequency of words in each class + vocab length (add-1 smoothing)
        for wordIndex in range(n_words):
            likelihood[wordIndex][0] /= (negNum + n_words)
            likelihood[wordIndex][1] /= (posNum + n_words)
        
        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params