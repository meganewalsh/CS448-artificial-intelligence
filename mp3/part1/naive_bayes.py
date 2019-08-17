import numpy as np

class NaiveBayes(object):
    def __init__(self,num_class,feature_dim,num_value):
        """Initialize a naive bayes model.

        This function will initialize prior and likelihood, where
        prior is P(class) with a dimension of (# of class,)
            that estimates the empirical frequencies of different classes in the training set.
        likelihood is P(F_i = f | class) with a dimension of
            (# of features/pixels per image, # of possible values per pixel, # of class),
            that computes the probability of every pixel location i being value f for every class label.

        Args:
            num_class(int): number of classes to classify // 10 Types of clothing
            feature_dim(int): feature dimension for each example
            num_value(int): number of possible values for each pixel
        """

        self.num_value = num_value
        self.num_class = num_class
        self.feature_dim = feature_dim

        self.prior = np.zeros((num_class))
        self.likelihood = np.zeros((feature_dim, num_value, num_class))

    def train(self,train_set,train_label):
        """ Train naive bayes model (self.prior and self.likelihood) with training dataset.
            self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
            self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of
                (# of features/pixels per image, # of possible values per pixel, # of class).
            You should apply Laplace smoothing to compute the likelihood.

        Args:
            train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
            train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
        """

        index_train_label = 0
        for picture in train_set:
            label = train_label[index_train_label]
            self.prior[label] += 1
            pixel = 0
            for color in picture:
                self.likelihood[pixel][color][label] += 1 
                pixel += 1
            index_train_label += 1

        for label in range(self.num_class):
            self.likelihood[:,:,label] = np.log10(self.likelihood[:,:,label]+1) - np.log10(self.prior[label]+10)

        self.prior = np.where(self.prior != 0, np.log10(np.true_divide(self.prior, len(train_label))), float("-inf"))


    def test(self,test_set,test_label):
        """ Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
            by performing maximum a posteriori (MAP) classification.
            The accuracy is computed as the average of correctness
            by comparing between predicted label and true label.

        Args:
            test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
            test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

        Returns:
            accuracy(float): average accuracy value
            pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
        """

        pred_label = np.zeros(len(test_set))
        most_common_label = np.zeros(self.feature_dim, dtype = int)

        for image, image_idx in zip(test_set, range(len(test_label))):
            max_value = float("-inf")
            max_label = -1
            for label in range(self.num_class):
                probability_label = self.prior[label]
                for pixel, color in zip(range(self.feature_dim), image):
                    probability_label += self.likelihood[pixel][color][label]
                if probability_label > max_value:
                    max_value = probability_label
                    max_label = label
            pred_label[image_idx] = max_label
     
        print("pred: ", pred_label) 
        accuracy = (np.sum(pred_label == test_label))/len(pred_label)
        print("Accuracy: ", accuracy)
        return accuracy, pred_label


    def save_model(self, prior, likelihood):
        """ Save the trained model parameters
        """
        np.save(prior, self.prior)
        np.save(likelihood, self.likelihood)

    def load_model(self, prior, likelihood):
        """ Load the trained model parameters
        """
        self.prior = np.load(prior)
        self.likelihood = np.load(likelihood)

    def intensity_feature_likelihoods(self, likelihood):
        """
        Get the feature likelihoods for high intensity pixels for each of the classes,
            by sum the probabilities of the top 128 intensities at each pixel location,
            sum k<-128:255 P(F_i = k | c).
            This helps generate visualization of trained likelihood images.

        Args:
            likelihood(numpy.ndarray): likelihood (in log) with a dimension of
                (# of features/pixels per image, # of possible values per pixel, # of class)
        Returns:
            feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
                (# of features/pixels per image, # of class)
        """
        p_fi = np.zeros((self.feature_dim, self.num_class))

        for pixel in range(self.feature_dim):
            for label in range(self.num_class):
                for color in range(128, self.num_value):
                    p_fi[pixel][label] += likelihood[pixel][color][label]

        p_fi = np.true_divide(p_fi, 128)
        return p_fi
