import unittest
import numpy as np
from naive_bayes import naive_bayes_predict,naive_bayes,predict_on_test_set,get_pos_neg_ratio,lookup

class TestNaiveBayes(unittest.TestCase):
    """Tests for methods used in Naive Bayes sentiment classification
    """
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        # initializr variables to use in testing the functions from naive_bayes file
        self.test_x=["am done :("]
        self.freqs={('am',1):1,('am',0):1,(':(',0):1,(':(',1):0,('happi',1):1,('happi',0):0}
        self.train_x=["am done :(","am happi"]
        self.train_y=[0,1]
        self.test_y=np.matrix([0.0])
        self.logprior,self.loglikelihood=0.0, {'am': 0.0, ':(': -0.6931471805599453, 'happi': 0.6931471805599453}
        

    def test_naive_bayes(self):
        """Test naive_bayes function 
        """
        self.assertEqual( naive_bayes(self.freqs,self.train_x,self.train_y),(0.0, {'am': 0.0, ':(': -0.6931471805599453, 'happi': 0.6931471805599453}))
    
    def test_naive_bayes_predict(self):
        """Test the naive_bayes_predict function
        """
        self.assertIsInstance(naive_bayes_predict(":(",self.logprior,self.loglikelihood),float)
        self.assertAlmostEqual(naive_bayes_predict(":(",self.logprior,self.loglikelihood),-0.69314718)
    
    def test_predict_on_test_set(self):
        """Test predict_on_test_set fucntion
        """
        

        self.assertEqual(predict_on_test_set(self.test_x,self.test_y,self.logprior,self.loglikelihood),100)
        pass
    def test_get_pos_neg_ratio(self):
        """Test the get_pos_neg_ratio function from naive_bayes
        """
        
        self.assertIsInstance(get_pos_neg_ratio(self.freqs),dict)
        self.assertDictEqual(get_pos_neg_ratio(self.freqs),{'am':['am',1,1,1],':(':[':(',0,1,0.5],'happi':['happi',1,0,2]})

    def test_lookup(self):
        """Test the lookup function
        """

        self.assertEqual(lookup(self.freqs,'am',0),1.0)
        self.assertEqual(lookup(self.freqs,'am',1),1.0)
        self.assertEqual(lookup(self.freqs,'happi',0),0)
        self.assertEqual(lookup(self.freqs,'happi',1),1.0)
        self.assertEqual(lookup(self.freqs,':(',0),1.0)
        self.assertEqual(lookup(self.freqs,':(',1),0)
        self.assertEqual(lookup(self.freqs,'done',1),0)
        self.assertEqual(lookup(self.freqs,'done',0),0)

if __name__ == '__main__':
    unittest.main()
        


        