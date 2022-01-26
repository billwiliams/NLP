import unittest
import numpy as np
from word_vectors import cosine_similarity, accuracy


class TestWordVectors(unittest.TestCase):
    """word vector tests

    """
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
    
    def test_cosine_similarity(self):
        A=np.random.rand(1,1)
        B=np.random.randint(1,size=1)
        
        self.assertEqual(cosine_similarity(A,A)[0][0],1.0)
        # self.assertEqual(cosine_similarity(vector,vector),0.0)

    
if __name__=='__main__':
    unittest.main()