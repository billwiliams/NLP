# Naive Bayes

Using Naive Bayes for sentiment analysis on tweets. Given a tweet,  decide if it has a positive sentiment or a negative one. Below are the guidelines: 

* Train a naive bayes model on a sentiment analysis task
* Test using the model
* Compute ratios of positive words to negative words
* Error analysis
* Predict on tweets

# Training model using Naive Bayes

Naive bayes is an algorithm that could be used for sentiment analysis. It takes a short time to train and also has a short prediction time.

#### Training a Naive Bayes classifier
- The first part of training a naive bayes classifier is to identify the number of classes that are available.

- We create a probability for each class.
$P(D_{pos})$ is the probability that the document is positive.
$P(D_{neg})$ is the probability that the document is negative.
Using the formulas as follows and store the values in a dictionary:

$$P(D_{pos}) = \frac{D_{pos}}{D}\tag{1}$$

$$P(D_{neg}) = \frac{D_{neg}}{D}\tag{2}$$

Where $D$ is the total number of documents, or tweets in this case, $D_{pos}$ is the total number of positive tweets and $D_{neg}$ is the total number of negative tweets.

#### Prior and Logprior

The prior probability represents the underlying probability in the target population that a tweet is positive versus negative.  In other words, if we had no specific information and blindly picked a tweet out of the population set, what is the probability that it will be positive versus that it will be negative? That is the "prior".

The prior is the ratio of the probabilities $\frac{P(D_{pos})}{P(D_{neg})}$.

We can take the log of the prior to rescale it, and we'll call this the logprior.

$$\text{logprior} = log \left( \frac{P(D_{pos})}{P(D_{neg})} \right) = log \left( \frac{D_{pos}}{D_{neg}} \right) $$



Note that $log(\frac{A}{B})$ is the same as $log(A) - log(B)$.  So the logprior can also be calculated as the difference between two logs:

$$\text{logprior} = \log (P(D_{pos})) - \log (P(D_{neg})) = \log (D_{pos}) - \log (D_{neg})\tag{3}$$

#### Positive and Negative Probability of a Word
To compute the positive probability and the negative probability for a specific word in the vocabulary, we use the following inputs:

- $freq_{pos}$ and $freq_{neg}$ are the frequencies of that specific word in the positive or negative class. In other words, the positive frequency of a word is the number of times the word is counted with the label of 1.
- $N_{pos}$ and $N_{neg}$ are the total number of positive and negative words for all documents (for all tweets), respectively.
- $V$ is the number of unique words in the entire set of documents, for all classes, whether positive or negative.

We use these to compute the positive and negative probability for a specific word using this formula:

$$ P(W_{pos}) = \frac{freq_{pos} + 1}{N_{pos} + V}\tag{4} $$
$$ P(W_{neg}) = \frac{freq_{neg} + 1}{N_{neg} + V}\tag{5} $$

Notice that we add the "+1" in the numerator for additive smoothing.  This [wiki article](https://en.wikipedia.org/wiki/Additive_smoothing) explains more about additive smoothing.

#### Computing naive Bayes
Given a freqs dictionary, `train_x` (a list of tweets) and a `train_y` (a list of labels for each tweet), We implement a naive bayes classifier as follows.

##### Calculate $V$
- We compute the number of unique words that appear in the `freqs` dictionary to get your $V$ 

##### Calculate $freq_{pos}$ and $freq_{neg}$
- Using  `freqs` dictionary, we  can compute the positive and negative frequency of each word $freq_{pos}$ and $freq_{neg}$.

##### Calculate $N_{pos}$, and $N_{neg}$
- Using `freqs` dictionary, we can also compute the total number of positive words and total number of negative words $N_{pos}$ and $N_{neg}$.

##### Calculate $D$, $D_{pos}$, $D_{neg}$
- Using the `train_y` input list of labels, we calculate the number of documents (tweets) $D$, as well as the number of positive documents (tweets) $D_{pos}$ and number of negative documents (tweets) $D_{neg}$.
- We Calculate the probability that a document (tweet) is positive $P(D_{pos})$, and the probability that a document (tweet) is negative $P(D_{neg})$

##### Calculating the logprior
- the logprior is $log(D_{pos}) - log(D_{neg})$

##### Calculating log likelihood
- Finally, we  can iterate over each word in the vocabulary,  to get the positive frequencies, $freq_{pos}$, and the negative frequencies, $freq_{neg}$, for that specific word.
- We compute the positive probability of each word $P(W_{pos})$, negative probability of each word $P(W_{neg})$ using equations 4 & 5.

$$ P(W_{pos}) = \frac{freq_{pos} + 1}{N_{pos} + V}\tag{4} $$
$$ P(W_{neg}) = \frac{freq_{neg} + 1}{N_{neg} + V}\tag{5} $$

**Note:** We use a dictionary to store the log likelihoods for each word.  The key is the word, the value is the log likelihood of that word).

- We can then compute the loglikelihood: $log \left( \frac{P(W_{pos})}{P(W_{neg})} \right)$.

# Testing  naive bayes

After computing `logprior` and `loglikelihood`, we can test the naive bayes function by making predicting on some tweets!

The `naive_bayes_predict` function makes predictions on tweets.
* The function takes in the `tweet`, `logprior`, `loglikelihood`.
* It returns the probability that the tweet belongs to the positive or negative class.
* For each tweet, we sum up loglikelihoods of each word in the tweet.
* Also add the logprior to this sum to get the predicted sentiment of that tweet.

$$ p = logprior + \sum_i^N (loglikelihood_i)$$

#### Note
We calculate the prior from the training data, and that the training data is evenly split between positive and negative labels (4000 positive and 4000 negative tweets).  This means that the ratio of positive to negative 1, and the logprior is 0.

The value of 0.0 means that when we add the logprior to the log likelihood, we're just adding zero to the log likelihood.  However,  whenever the data is not perfectly balanced, the logprior will be a non-zero value.

#  Error Analysis

Checking some tweets that the model missclassified. 
- @jaredNOTsubway @iluvmariah @Bravotv Then that truly is a LATERAL move! Now, we all know the Queen Bee is UPWARD BOUND : ) #MovingOnUp

    predicted 0

    correct label [1.]

- A new report talks about how we burn more calories in the cold, because we work harder to warm up. Feel any better about the weather? :p

    predicted 0

    correct label [1.]

- Harry and niall and -94 (when harry was born) ik it's stupid and i wanna change it :D https://t.co/gHAt8ZDAfF

    predicted 0

    correct label [1.]

- off to the park to get some sunlight : )

    predicted 0

    correct label [1.]

- @msarosh Uff Itna Miss karhy thy ap :p

    predicted 0
    correct label [1.]

- @rcdlccom hello, any info about possible interest in Jonathas ?? He is close to join Betis :( greatings

    predicted 1

    correct label [0.]

-   @phenomyoutube u probs had more fun with david than me : (

  predicted 1
    correct label [0.]

- pats jay : (
    predicted 1
    correct label [0.]

- Sr. Financial Analyst - Expedia, Inc.: (#Bellevue, WA) http://t.co/ktknMhvwCI #Finance #ExpediaJobs #Job #Jobs #Hiring

    predicted 1

    correct label [0.]





# Filtering words by Ratio of positive to negative counts

- Some words have more positive counts than others, and can be considered "more positive".  Likewise, some words can be considered more negative than others.
- One way  to define the level of positiveness or negativeness, without calculating the log likelihood, is to compare the positive to negative frequency of the word.
    - Note that we can also use the log likelihood calculations to compare relative positivity or negativity of words.
- We can calculate the ratio of positive to negative frequencies of a word.
- Once we're able to calculate these ratios, we can also filter a subset of words that have a minimum ratio of positivity / negativity or higher.
- Similarly, we can also filter a subset of words that have a maximum ratio of positivity / negativity or lower (words that are at least as negative, or even more negative than a given threshold).

#### **More Negative words (threshold of less  0.05)**
    :( ratio 0.000544069640914037

    :-( ratio 0.002583979328165375

    zayniscomingbackonjuli ratio 0.05

    26 ratio 0.047619047619047616

    >:( ratio 0.022727272727272728

    lost ratio 0.05

    ♛ ratio 0.004739336492890996

    》 ratio 0.004739336492890996

    beli̇ev ratio 0.027777777777777776

    wi̇ll ratio 0.027777777777777776

    justi̇n ratio 0.027777777777777776

    ｓｅｅ ratio 0.027777777777777776

    ｍｅ ratio 0.027777777777777776

#### **More positive words threshold of (8.0)**
    followfriday ratio 24.0
    engag ratio 8.0
    commun ratio 14.0
    :) ratio 987.0
    flipkartfashionfriday ratio 17.0
    happi ratio 8.578947368421053
    friday ratio 9.2
    :d ratio 524.0
    :p ratio 106.0
    influenc ratio 17.0
    great ratio 8.5
    opportun ratio 9.0
    :-) ratio 553.0
    here' ratio 21.0
    youth ratio 15.0
    bam ratio 45.0
    warsaw ratio 45.0
    shout ratio 12.0
    twitch ratio 8.0
    ;) ratio 23.0
    welcom ratio 9.166666666666666
    stat ratio 52.0
    arriv ratio 11.6
    via ratio 8.222222222222221
    appreci ratio 9.666666666666666
    invit ratio 8.0
    glad ratio 14.0
    blog ratio 28.0
    fav ratio 12.0
    goodnight ratio 9.5
    vid ratio 9.0
    fantast ratio 10.0
    braindot ratio 9.0
    men ratio 9.0
    ff ratio 8.2
    recent ratio 9.0
    fback ratio 27.0
    goodmorn ratio 8.0
    spread ratio 8.0
    ceo ratio 9.0
    1month ratio 9.0
    follback ratio 9.5
    hehe ratio 9.0
    wick ratio 9.0
    earth ratio 8.0
    pleasur ratio 11.0
    minecraft ratio 8.0
    💙 ratio 9.0
    ← ratio 10.0
    aqui ratio 10.0

