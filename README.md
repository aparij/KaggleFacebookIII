KaggleFacebookIII
=================
http://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction


Very late entry to Kaggle competition.Predicting tags to StackExchange questions.
My final standing was quite low 316/380, but I just had 3 weeks comparing to 4 months of the competition.
It was my first experience with Gensim library. 
I tried to model topics of the questions using LDA. For preprocessing stemming , punctuation marks removed. 
Less frequent words too(words that I encountered only once) . Tried to remove code snippets too.
Then I would find the most similar question based on topics and take the tags from training to the test question
and drop less frequent tags

So what could have I done better ?
Preprocessing should be done once, I did every time I was building dictionary or corpora . It would probably speed 
up by at least x2. 
More automatic parallezation , although I used MLK to get free optimization I could have made some simple code to 
load split test file. Althought now it can be done just by using linux split on test.csv and then combining the results.




Steps:

1. Build dictionary
2. Build corpara
3. Build model
4. Build similarity matrix
5. Find similarities between test and the corpora






Requires:

Gensim, NLTK, numpy 

Nice to have
MLK,Atlas.BLAS or any other linear algebra optimization libraries
