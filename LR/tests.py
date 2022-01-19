import unittest


from LR import sigmoid,extract_features
from utils import build_features,process_tweet

class TestLogisticRegression(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.train_x=["am done :(", "winning happiness"]
        self.train_y=[[0],[1]]
        self.test_x=["sad :("]
        self.test_y=[0]  
        self.freqs={('done',0):1,(':(',0):1,('win',1):1,('happi',1):1}  
   

    # tests for the functions

    def test_build_features(self):
        self.assertEqual(build_features(self.train_x,self.train_y),self.freqs)

    def test_process_tweet(self):

        self.assertEqual(process_tweet(self.train_x[0]),['done', ':('])
        self.assertEqual(process_tweet(self.train_x[1]),['win', 'happi'])

        

    def test_sigmoid(self):
        self.assertEqual(sigmoid(0),0.5)

    def test_extract_features(self):
        """Test extract features function
        """

        self.assertEqual(extract_features("am done",self.freqs),[1,1,0])

    def test_predict(self):
        """tests the predict function

        Args:
            predict ([function]): [function to predict sentiment]
        """
        pass

if __name__=='__main__':
    unittest.main()
