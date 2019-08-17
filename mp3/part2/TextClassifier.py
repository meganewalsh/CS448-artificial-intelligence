# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

import math

# MIN_INT = -9223372036854775807

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0
        self.p_class = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}
        self.p_word_class = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}, 9:{}, 10:{}, 11:{}, 12:{}, 13:{}, 14:{}}
        self.p_bigram_class = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}, 9:{}, 10:{}, 11:{}, 12:{}, 13:{}, 14:{}}

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        # TODO: Write your code here
        total_distinct_words = set()
        total_distinct_bigrams = set()
        total_text_count = 0
        total_words_per_class = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}
        total_bigrams_per_class = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}
        for text, clas in zip(train_set, train_label):
            total_text_count += 1
            self.p_class[clas] += 1
            total_word_count = 0
            curr_class_dict = self.p_word_class[clas]
            for word in text:
                total_distinct_words.add(word)
                total_word_count += 1
                if word in curr_class_dict:
                    curr_class_dict[word] += 1
                else:
                    curr_class_dict[word] = 1
            total_words_per_class[clas] += total_word_count

            total_bigram = 0
            curr_class_dict = self.p_bigram_class[clas]
            for i in range(1, len(text)):
                total_bigram += 1
                bigram = (text[i-1], text[i])
                total_distinct_bigrams.add(bigram)
                if bigram in curr_class_dict:
                    curr_class_dict[bigram] += 1
                else:
                    curr_class_dict[bigram] = 1
            total_bigrams_per_class[clas] += total_bigram

        # print(self.p_word_class[1])
        # print(total_words_per_class)

        for clas in range(1, 15):
            if self.p_class[clas] != 0:
                self.p_class[clas] = math.log(self.p_class[clas]) - math.log(total_text_count)
            else:
                self.p_class[clas] = float("-inf")
            curr_class_dict = self.p_word_class[clas]
            total_words = total_words_per_class[clas]
            # distinct_words = len(curr_class_dict)
            distinct_words = len(total_distinct_words)
            # for word in curr_class_dict:
            for word in total_distinct_words:
                if word in curr_class_dict:
                    curr_class_dict[word] = math.log(curr_class_dict[word] + 0.1) - math.log(total_words + distinct_words*0.1)
                else:
                    curr_class_dict[word] = math.log(0 + 0.1) - math.log(total_words + distinct_words*0.1)

            curr_class_dict = self.p_bigram_class[clas]
            total_b = total_bigrams_per_class[clas]
            # distinct_b = len(curr_class_dict)
            distinct_b = len(total_distinct_bigrams)
            # for b in curr_class_dict:
            for b in total_distinct_bigrams:
                if b in curr_class_dict:
                    curr_class_dict[b] = math.log(curr_class_dict[b] + 0.1) - math.log(total_b + distinct_b*0.1)
                else:
                    curr_class_dict[b] = math.log(0 + 0.1) - math.log(total_b + distinct_b*0.1)

        # print(self.p_word_class[1])


    def predict(self, dev_set, dev_label, lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """

        accuracy = 0.0
        result = []

        # TODO: Write your code here
        total = 0
        for text in dev_set:
            total += 1
            min_prob = float("-inf")
            guess = 0

            for clas in range(1, 15):
                sum_prob = 0
                curr_class_dict = self.p_word_class[clas]
                for word in set(text):
                    if word in curr_class_dict:
                        sum_prob += curr_class_dict[word]
                    # else:
                    #     sum_prob += -500#-9.4
                prob_uni = self.p_class[clas]+sum_prob

                sum_b = 0
                curr_class_dict = self.p_bigram_class[clas]
                for i in range(1, len(text)):
                    bigram = (text[i-1], text[i])
                    if bigram in curr_class_dict:
                        sum_b += curr_class_dict[bigram]
                    # else:
                    #     sum_b += -100
                prob_bi = self.p_class[clas]+sum_b

                prob_class = (1-lambda_mix)*prob_uni + lambda_mix*prob_bi
                if prob_class > min_prob:
                    min_prob = prob_class
                    guess = clas

            result.append(guess)

        correct = 0
        for clas, guess in zip(dev_label, result):
            if clas == guess:
                correct += 1

        accuracy = correct / total

        return accuracy,result

    def top_20(self):
        ret = []
        for key in range(1, 15):
            words = self.p_word_class[key]
            top_20 = []
            for i in range(0, 20):
                max = float("-inf")
                max_word = ''
                for word in words.keys():
                    if words[word] > max:
                        max = words[word]
                        max_word = word
                top_20.append((max_word, max))
                words.pop(max_word, None)
            ret.append(top_20[:])
            print("Class ", key)
            print(top_20)
