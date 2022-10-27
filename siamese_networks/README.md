### Siamese Networks
cheking if two questions are the same using siamese networks

# Part 1: Importing the Data
<a name='1.1'></a>
### 1.1 Loading in the data

We will be using the Quora question answer dataset to build a model that could identify similar questions. This is a useful task because we don't want to have several versions of the same question posted.
we select only the question pairs that are duplicate to train the model. <br>
We build two batches as input for the Siamese network and we assume that question $q1_i$ (question $i$ in the first batch) is a duplicate of $q2_i$ (question $i$ in the second batch), but all other questions in the second batch are not duplicates of $q1_i$. 

TRAINING QUESTIONS:

Question 1:  Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?
Question 2:  I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me? 

Question 1:  What would a Trump presidency mean for current international master’s students on an F1 visa?
Question 2:  How will a Trump presidency affect the students presently in US or planning to study in US? 

TESTING QUESTIONS:

Question 1:  How do I prepare for interviews for cse?
Question 2:  What is the best way to prepare for cse? 

is_duplicate = 0   
