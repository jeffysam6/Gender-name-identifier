import nltk
from nltk.corpus import names
import random



def gender_feature(name):
	feature = {"last_char" : name[-1]}
	return feature


labeled_name = ([(name,'male') for name in names.words("male.txt")] + [(name,'female') for name in names.words("female.txt")])

random.shuffle(labeled_name)


featurelist = [(gender_feature(i),gender) for (i,gender) in labeled_name]

"""featurelist[0]
({'last_char': 'e'}, 'female')"""


train_list,test_list = featurelist[:1000],featurelist[1000:2000]

classifiering = nltk.NaiveBayesClassifier.train(train_list)
print(classifiering.classify(gender_feature("kumar")))
#'male'
print(classifiering.classify(gender_feature("harold")))
#'male'