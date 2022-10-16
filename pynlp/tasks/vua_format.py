import pandas as pd
from collections import Counter

#Import extra Data
#fields =['acl.id','text','label']
#extra_data = pd.read_csv('data/gibert/extraData2.csv', delimiter="\t")




class VuaFormat:
    """VUA-format data"""

    def __init__(self):
        self.training_file = 'train.csv'
        # self.extra_file = 'extraData2.csv'
        self.task = None
        self.name = "VUA_format"
        self.train_data = None
        self.test_data = None

    def __str__(self):
        return self.name + ", " + self.task

    def load(self, data_dir, test_file='olid-test.csv'):
        """"loads training and test data"""

        self.train_data = pd.read_csv(data_dir + 'train.csv', delimiter=",")
        self.test_data = pd.read_csv(data_dir + test_file, delimiter=",")


    def train_instances(self):
        """returns training instances and labels for a given task

        :return: X_train, y_train
        """
        
        return self.train_data['text'], self.train_data['labels']

    def test_instances(self):
        return self.test_data['text'], self.test_data['labels']
        
