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

  I build recommendation system in this project for this solution.
  
**What Will These Solutions Bring to The Company?:**

- easy product comparison
- defining like/dislikes easily 
- saving time
- more money with selling more products 
- happier customers = more customers = more money
- less time on server = less problem

**Data:**

In this project, I worked on sentiment analysis of Kindle Store reviews in Amazon. I choose this dataset because it is more easy to buy and read a book with Kindle. Going to the book store, finding a book which you like need more time than reaching every book from your tablet. 

The data is obtained from github.io page of [UC San Diego Computer Science and Engineering Department academic staff](https://nijianmo.github.io/amazon/index.html#subsets). Dataset contains product reviews and metadata, including 142.8 million reviews from May 1996 to July 2014. I prefer to use 5-core sample dataset(such that each of the remaining users and items have at least 5 reviews) and metadata for Kindle Store. The reasons to choose 5-core data is that continuous users contains more information than single reviewers. To reach and download metadata, people have to fill the form and submit it. My filtered Kindle Store data consists of 2,222,983 rows and 12 columns. Also, I used the metadata to find the corresponding titles of the books from product ID.

**Plan:**

- Understanding, Cleaning and Exploring Data: To analyze distributions of data points, I observed each column seperately and compared common words in positive, negative and neutral reviews. The first challange of this data is to clean text from unnecessary items for modeling such as punctuation, upper-case letters etc. Detailed data analysis can be found [here](https://github.com/ezgigm/sentiment_analysis_and_product_recommendation/blob/master/notebooks/2_Understanding_EDA_Preparation.ipynb)

- 
**Findings:**

**Future Improvements:**

 # Repository Guide
 
  **CSV Files:**
  
  **Notebooks:**
  
  **Presentation:**
  
  # Resources 
