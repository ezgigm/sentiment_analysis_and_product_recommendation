# Sentiment Analysis and Product Recommendation from Reviews
From Kindle Store Reviews on Amazon, sentiment analysis and book recommendation

**Problem:**

Day by day, demand in e-commerce is increasing. With the increasing demand in online stores, voice of customer concepts such as reviews and customer experiences are getting more important. Because, customers buy products without seeing or touching them. If the company fails to meet this need of customers, it loses money because of not taking strategic decisions.

**Aim:** 

Protect company from losing money with increasing customer satisfaction and giving importance to their feedback.

**Solution:**

- define good or bad products as quick as possible according to reviews and take action for this

For this solution, I worked on sentiment analysis with different models. The model predicts reviews as positive or negative from text.

- recommend customers related products to increase satisfaction with decreasing search time for suitable product

I built recommendation system in this project for this solution.
  
**What Will These Solutions Bring to The Company?**

- easy product comparison
- defining dislikes easily 
- saving time
- more money with selling more products 
- happier customers = more customers = more money
- less time on server = less problem

**Data:**

In this project, I worked on sentiment analysis of Kindle Store reviews in Amazon. I choose this dataset because it is more easy to buy and read a book with Kindle. Going to the book store, finding a book which you like need more time than reaching every book from your tablet. 

The data is obtained from github.io page of [UC San Diego Computer Science and Engineering Department academic staff](https://nijianmo.github.io/amazon/index.html#subsets). Dataset contains product reviews and metadata, including 142.8 million reviews from May 1996 to July 2014. I prefer to use 5-core sample dataset(such that each of the remaining users and items have at least 5 reviews) and metadata for Kindle Store. The reasons to choose 5-core data is that continuous users contains more information than single reviewers. To reach and download metadata, people have to fill the form and submit it. My filtered Kindle Store data consists of 2,222,983 rows and 12 columns. Also, I used the metadata to find the corresponding titles of the books from product ID. The format of raw data is json. 

**Plan:**

- ***Understanding, Cleaning and Exploring Data:*** To analyze distributions of data points, I observed each column seperately and compared common words in positive, negative and neutral reviews. The first challange of this data is to clean text from unnecessary items for modeling such as punctuation, upper-case letters etc. Detailed data analysis can be found [here](https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/2_Understanding_EDA_Preparation.ipynb).

- ***Preparing Data to Modeling:*** Target was changed to binary class. Machine learning models and neural net models have different preparing strategies. Mainly, vectorization/tokenization, spliting train-test sets and padding were done. Detailed pre-processing techniques and steps can be found in corresponding notebooks. 

- ***Modeling:*** Firstly, LogReg, DecisionTree, Extra-Trees, RandomForest, XGBoost and LGBM Classifiers were tried. Then, FastText class of Torch models were tried with different parameters. Keras models as CNN with 3 convolutional layers, RNN with 2 GRU layer, RNN with 2 LSTM layers, RNN with 2 CuRNNGRU layers and CNN with 2 convolutional layers were built. At last, pre-trained BERT model was tried. 

- ***Evaluation and Results:*** To compare my results, I used balanced accuracy for machine learning models and loss values for deep learning models. I also calculated accuracy values for neural nets to represent my results in smart way. Although the maximum accuracy between machine learning models is 87% for test set with LogReg, it is 95% between deep learning model with pre-trained BertForSequence Classifier from BERT. It means that my model can predict the sentiment of review as positive or negative with 95% accuracy. 

- ***Recommendation Systems:*** There different system were established. One is collaborative filtering with matrix factorization(SVDS) , second one is cosine-similarity of user-user based. As last one, I tried to solve cold-start problem with taking a few information from new user such as keywords. As a different approach, without looking summaries or genres of books, recommendations were done by the cosine-similarity of keywords and reviews. To this last system, rating effect, rating number effect and positive rating effect were added orderly and scores were compared.

Resources were added to corresponding notebooks. If I was inspired by some external resources such as models, ideas etc, I inserted them corresponding notebooks.

**Findings:**

- My target is highly imbalanced but neural nets are very good at solving this issue. 
- Most of the data points are belongs to recent years especially after 2014.
- Data contains balanced points from each date and month. Also, distribution of target classes are similar in each time period. 4 and 5 rated books are on majority for all periods. 
- High review average does not mean the book is better than others. It is more accurate way to look review numbers. It is hard to say that book with 5 average rating with 8 reviews is better than the book with 4.3 average rating with 2000 reviews. 
- Although some top common words are similar in negative and positive reviews, some of the, totally different.
- Tranfer learning is easier than build a model and train it from zero level.
- Deep-learning models can handle overfitting problem better than machine learning models. 
- Results change when the layer types and layer numbers change.
- Bi-gram , tri-gram feature engineering effect results.
- When the type of recommendation system changes, recommendations also change. 
- When more data is added to the recommendation system, score increases. 

More findings for data can be found [here](https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/2_Understanding_EDA_Preparation.ipynb), and different findings for each model can be found in corresponding notebooks.

**Future Improvements:**

Each model has different improvements and they can be found in notebooks. Here, I will state about BERT model (determined as giving best results) and recommendation system improvements here. 

- Batch and epoch numbers can be tuned better way for modeling.
- Learning rate and epsilon values can be changed for modeling.
- More data can be added to recommendation systems.
- Positive review numbers ratio to negative review numbers effect can be added to recommendation function.

 # Repository Guide
 
 **CSV Files:**
 
 The sample data was downloaded to this repo; https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/sample_data.csv
 
 **Notebooks:**
 
There are total 9 notebooks in this repo. All of them was collected in notebooks file. For details;

Getting data from json file: https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/1_Getting_Data_from_json_Files%20.ipynb
 
More information about importance of sentiment analysis, every steps for data understanding, cleaning, EDA ; https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/2_Understanding_EDA_Preparation.ipynb

Machine learning models with metric: https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/3_Machine_Learning_Models.ipynb

Torch models with setting different parameters: https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/4_Torch_Models.ipynb

Keras models for 3-convolutional CNN and 2-GRU layers RNN: https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/5_Keras_CNN_with_3_Conv_Layers_and_RNN_with_2_GRU_Layers.ipynb

Keras models with different layer numbers and types and comparison of results for all Keras models:
https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/6_Keras_with_Different_Layer_Types_and_Numbers.ipynb

Pre-trained Bert Model: https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/7_Pre_trained_BERT_model.ipynb

Recommendation systems for collaborative filtering with matrix factorization and cosine similarity: https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/8_Recommendation_Systems.ipynb

Recommendations systems from cosine similarity of keywords with reviews: https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/9_Recommendation_from_Keywords.ipynb
 
 **Presentation:**
 
 Presentation can be found here in .pdf format ;
 
 **Video:**
 
 The video of presentation is [here]().
 
 **Reproduction:**
 
 - Clone this repo (for help see this [tutorial](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
 - Sample data is available in this repo, notebooks can be run with this sample data.
 
 ***Contact:*** [Ezgi Gumusbas](https://www.linkedin.com/in/ezgi-gumusbas-6b08a51a0/)
