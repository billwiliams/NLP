import unittest
import numpy as np
from word_vectors import cosine_similarity, accuracy


class TestWordVectors(unittest.TestCase):
    """word vector tests

    """
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
    
    def test_cosine_similarity(self):
        vector=np.random.rand(3,3)
        print(cosine_similarity(vector,vector))
        # self.assertEqual(cosine_similarity(vector,vector),1.0)
        # self.assertEqual(cosine_similarity(vector,vector),0.0)

    
if __name__=='__main__':
    unittest.main()