import unittest
import numpy as np
from naive_bayes import naive_bayes_predict,naive_bayes,predict_on_test_set

class TestNaiveBayes(unittest.TestCase):
    """Tests for methods used in Naive Bayes sentiment classification
    """
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.test_x=["am done :("]
        self.freqs={('am',1):1,('am',0):1,(':(',0):1,(':(',1):0,('happi',1):1,('happi',0):0}
        self.train_x=["am done :(","am happi"]
        self.train_y=[0,1]
        self.test_y=np.matrix([0])
        self.logprior,self.loglikelihood=0.0, {'am': 0.0, ':(': -0.6931471805599453, 'happi': 0.6931471805599453}
        

    def test_naive_bayes(self):
        self.assertEqual( naive_bayes(self.freqs,self.train_x,self.train_y),(0.0, {'am': 0.0, ':(': -0.6931471805599453, 'happi': 0.6931471805599453}))
    def test_naive_bayes_predict(self):
        self.assertIsInstance(naive_bayes_predict("am done",self.logprior,self.loglikelihood),float)
    def test_predict_on_test_set(self):
        

        # self.assertEqual(predict_on_test_set(self.test_x,self.test_y.flatten(),self.logprior,self.loglikelihood),100)
        pass

if __name__ == '__main__':
    unittest.main()
        


        