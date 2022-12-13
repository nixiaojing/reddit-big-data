# Databricks notebook source
# MAGIC %md
# MAGIC ### Read the joined NFL data in

# COMMAND ----------

nfl = spark.read.parquet("/FileStore/nfl_with_external/nfl_with_external.parquet")

# COMMAND ----------

tfidf_df = nfl.select('title','body')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Conduct basic data text checks / analysis on your data. What are the most common words overall or over time? What is the distribution of text lengths? What are important words according to TF-IDF?
# MAGIC 
# MAGIC #### Identify important keywords for your reddit data and use regex searches to create at least two dummy variables to identify comments of particular topics. -- 1. home/away, 2. 
# MAGIC 
# MAGIC #### Clean your text data using johnsnowlabs sparkNLP. Think about a few standard procedures to use: stop words, stemming, lemmatizing, removing unusual characters, matching synonyms, etc. You must use at least five NLP cleaning procedures.

# COMMAND ----------

# MAGIC %md
# MAGIC ### First, let's look at the length of the comments in our dataset
# MAGIC In this session, the comment length distribution is calculated. (split by space)

# COMMAND ----------

from pyspark.sql.functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
tfidf_df=tfidf_df.withColumn("comment_length", size(split(col('body'), ' ')))

# COMMAND ----------

tfidf_df.show()

# COMMAND ----------

import os
PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# COMMAND ----------

PLOT_DIR

# COMMAND ----------

## histogram (by groupby)
hist_comment_length = tfidf_df.groupBy('comment_length').count().toPandas()
hist_comment_length = hist_comment_length.sort_values('comment_length')
hist_comment_length

# COMMAND ----------

hist_comment_length['cum_count'] = hist_comment_length['count'].cumsum()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create a table contains cumulative comments count of every comment length

# COMMAND ----------

hist_comment_length

# COMMAND ----------

# save the results to csv
fpath = os.path.join(CSV_DIR, "hist_comment_length.csv")
hist_comment_length.to_csv(fpath)

# COMMAND ----------

## 90% cut
comment_length_90 = hist_comment_length.cum_count.iloc[-1]*0.9
x=hist_comment_length.cum_count<comment_length_90
ninty_cut = x.value_counts()
ninty_cut

# COMMAND ----------

## 90%
hist_comment_length.iloc[ninty_cut.iloc[1]]


# COMMAND ----------

hist_comment_length_90 = hist_comment_length.iloc[:ninty_cut.iloc[1]]

# COMMAND ----------

hist_comment_length_90.head(10)

# COMMAND ----------

hist_comment_length_90 = hist_comment_length.iloc[:ninty_cut.iloc[1]]

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(x[i],y[i]+20,y[i], ha = 'center',
                 Bbox = dict(facecolor = 'lightsalmon', alpha = .5))
        
fig = plt.figure(figsize = (22, 8))
plt.style.use('fivethirtyeight')
# creating the bar plot
plt.bar('comment_length', 'count', data=hist_comment_length_90, color='maroon') 
addlabels(hist_comment_length_90['comment_length'].tolist(), hist_comment_length_90['count'].tolist())
plt.title("Comment Length Histogram")
plt.xlabel("Comment Length")
plt.xlim([0.5, 47])
plt.ylabel("Comment Frequency")

## Save the plot in the plot dir so that it can be checked in into the repo
plot_fpath = os.path.join(PLOT_DIR, 'histogram_n_word_comment.png')
plt.savefig(plot_fpath)

plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### After calculating and counting the length of each comments, let's build a pipeline to clean our subreddit data to build a wordcloud

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build a pipieline for wordcloud

# COMMAND ----------

#%%time
document_assembler = DocumentAssembler()\
      .setInputCol("body")\
      .setOutputCol("document")
    
tokenizer = Tokenizer()\
      .setInputCols(["document"])\
      .setOutputCol("token")
      
normalizer = Normalizer()\
    .setInputCols(["token"])\
    .setOutputCol("normalized")\
    .setLowercase(True)\
    .setCleanupPatterns(["""[^\w\d\s]"""]) # remove punctuations (keep alphanumeric chars)
# if we don't set CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])

stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

finisher = Finisher() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)# To generate Term Frequency

# convert labels (string) to integers. Easy to process compared to string.
label_stringIdx = StringIndexer(inputCol = "title", outputCol = "label")

nlp_pipeline = Pipeline(stages=[document_assembler, 
            tokenizer,    
            normalizer,
            stopwords_cleaner,
            finisher,
#             hashingTF,
#             idf,
            label_stringIdx])
#             lr,
#             label_to_stringIdx])



# COMMAND ----------

# MAGIC %md
# MAGIC #### For the sake of better understanding each word in wordcloud, this wordcloud pipeline did not use stemmer function to lemmatize the words

# COMMAND ----------

nlp_model_wordcloud = nlp_pipeline.fit(tfidf_df)

processed_wordcloud = nlp_model_wordcloud.transform(tfidf_df)

# COMMAND ----------

words = processed_wordcloud.select('token_features').cache()

# COMMAND ----------

words.show(3)

# COMMAND ----------

from pyspark.sql.functions import concat_ws

words = words.withColumn("words", concat_ws(" ", "token_features")).cache()

# COMMAND ----------

words_df=words.select('words').toPandas()['words']
words_list=list(words_df)
words_text = ' '.join(words_list)
words_text

# COMMAND ----------

# MAGIC %md
# MAGIC #### Let's make a wordcloud of all the comments across the total 17 games

# COMMAND ----------

!pip install WordCloud

# COMMAND ----------

from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
 
#Create the mask
colosseum_mask = np.array(Image.open('image/nfl-league-logo.png'))
#Grab the mask colors
colors = ImageColorGenerator(colosseum_mask)
#Filter out stop words
stopwords = set(STOPWORDS)
wordcloud = WordCloud(mask=colosseum_mask,
                      scale=6,
                      include_numbers=True,
                      max_words=10000000,
                      color_func=colors,
                      background_color='white',
                      stopwords=stopwords,
                      collocations=True,
                      contour_color='#5d0f24', #013369
                      contour_width=3,
                      relative_scaling=0.5,
                      min_font_size=4, #colormap='Reds_r'
                      max_font_size=60).generate_from_text(words_text)
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis("off")
plt.title('All Post Game Thread Wordcloud', x=0.5, y=1, fontweight="bold", color="#d50a0a", size=30)

# Save the plot in the plot dir so that it can be checked in into the repo
plot_fpath = os.path.join(PLOT_DIR, 'all_post_game_thread_wordcloud.png')
plt.savefig(plot_fpath)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Let's visualize the total comments number in each post game thread

# COMMAND ----------

from pyspark.sql.functions import desc
processed_wordcloud.groupBy("title").count().sort(desc('count')).show(5,truncate=False)

# COMMAND ----------

comment_count = processed_wordcloud.groupBy("title").count().sort(desc('count')).cache().toPandas()

# COMMAND ----------

# save the results to csv
fpath = os.path.join(CSV_DIR, "all_post_game_thread_comments_count.csv")
comment_count.to_csv(fpath)

# COMMAND ----------

import matplotlib.pyplot as plt
from matplotlib import style
        
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(y[i]+30, x[i],y[i], ha = 'center',
                 Bbox = dict(facecolor = '#d50a0a', alpha = .7))
# Using the style for the plot
fig, ax = plt.subplots()
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [12, 10]
plt.barh('title','count', data=comment_count, color='#013369',alpha=0.8)
addlabels(comment_count['title'], comment_count['count'])
plt.ylabel("Subreddit NFL Teams")
plt.xlabel("Submission Count")
plt.title('Post Game Thread Comments Count')

# Save the plot in the plot dir so that it can be checked in into the repo
plot_fpath = os.path.join(PLOT_DIR, 'post_game_thread_comments_count.png')
plt.savefig(plot_fpath)
ax.invert_yaxis()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Since New York Giants and Washington Commanders' game had the highest amount of comments, let's make the wordcloud for that game

# COMMAND ----------

Giant_Commenders = processed_wordcloud.filter(col('title')=='Post Game Thread: New York Giants at Washington Football Team').cache()

# COMMAND ----------

Giant_Commenders.show(5)

# COMMAND ----------

words_Giant_Commenders = Giant_Commenders.select('token_features').cache()

# COMMAND ----------

words_Giant_Commenders.show(5)

# COMMAND ----------

from pyspark.sql.functions import concat_ws

words_Giant_Commenders = words_Giant_Commenders.withColumn("words", concat_ws(" ", "token_features")).cache()

# COMMAND ----------

words_df_GC=words_Giant_Commenders.select('words').toPandas()['words']
words_list_GC=list(words_df_GC)
words_text_GC = ' '.join(words_list_GC)
words_text_GC

# COMMAND ----------

!pip install WordCloud

# COMMAND ----------

from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
 
#Create the mask
colosseum_mask = np.array(Image.open('image/GC.png'))
#Grab the mask colors
colors = ImageColorGenerator(colosseum_mask)
#Filter out stop words
stopwords = set(STOPWORDS)
wordcloud = WordCloud(mask=colosseum_mask,
                      scale=6,
                      include_numbers=True,
                      max_words=10000000,
                      color_func=colors,
                      background_color='white',
                      stopwords=stopwords,
                      collocations=True,
                      contour_color='#5d0f24', #013369
                      contour_width=5,
                      relative_scaling=0.5,
                      min_font_size=4, #colormap='Reds_r'
                      max_font_size=70).generate_from_text(words_text_GC)
plt.figure(figsize=(18,15))
plt.imshow(wordcloud)
plt.axis("off")
plt.title('Giants at Commanders Post Game Thread Wordcloud', x=0.5, y=1, fontweight="bold", color="#d50a0a", size=30)

# Save the plot in the plot dir so that it can be checked in into the repo
plot_fpath = os.path.join(PLOT_DIR, 'giants_at_commanders_post_game_thread_wordcloud.png')
plt.savefig(plot_fpath)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now let's use our pipeline again with stemmer function to calculate TF-IDF of each word

# COMMAND ----------

# spark = SparkSession\
#         .builder\
#         .appName("TfIdf_Example")\
#         .getOrCreate()

# COMMAND ----------

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, StringIndexer, CountVectorizer

# COMMAND ----------


stemmer = Stemmer()\
      .setInputCols(["cleanTokens"])\
      .setOutputCol("stem")

finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# # To generate Term Frequency
# hashingTF = HashingTF(inputCol="token_features", outputCol="TF", numFeatures=1000)

# # To generate Inverse Document Frequency
# idf = IDF(inputCol="TF", outputCol="IDF", minDocFreq=5)

# convert labels (string) to integers. Easy to process compared to string.
label_stringIdx = StringIndexer(inputCol = "title", outputCol = "label")


# countVectors = CountVectorizer(inputCol="token_features", outputCol="features", vocabSize=10000, minDF=5)

# label_stringIdx = StringIndexer(inputCol = "features", outputCol = "label")

nlp_pipeline = Pipeline(stages=[document_assembler, 
            tokenizer,    
            normalizer,
            stopwords_cleaner,
            stemmer, 
            finisher,
#             hashingTF,
#             idf,
            label_stringIdx])
#             lr,
#             label_to_stringIdx])

nlp_model = nlp_pipeline.fit(tfidf_df)

processed = nlp_model.transform(tfidf_df)

processed.count()

# COMMAND ----------

processed.show()

# COMMAND ----------

clean_terms = processed.select('title','token_features')
clean_terms.show()

# COMMAND ----------

clean_terms_rdd = clean_terms.rdd

# COMMAND ----------

clean_terms_rdd.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Here we calculate TF-IDF for each word and document

# COMMAND ----------

import math
from pyspark.sql.functions import *

map1=clean_terms_rdd.flatMap(lambda x: [((x[0],i),1) for i in x[1]])
reduce=map1.reduceByKey(lambda x,y:x+y)
tf=reduce.map(lambda x: (x[0][1],(x[0][0],x[1])))
map3=reduce.map(lambda x: (x[0][1],(x[0][0],x[1],1)))
map4=map3.map(lambda x:(x[0],x[1][2]))
reduce2=map4.reduceByKey(lambda x,y:x+y)
idf=reduce2.map(lambda x: (x[0],math.log10(7511/x[1])))
rdd=tf.join(idf)
rdd=rdd.map(lambda x: (x[1][0][0],(x[0],x[1][0][1],x[1][1],x[1][0][1]*x[1][1]))).sortByKey()
rdd=rdd.map(lambda x: (x[0],x[1][0],x[1][1],x[1][2],x[1][3]))
rdd.toDF(["title","Token","TF","IDF","TF-IDF"]).show()

# COMMAND ----------

tfidf_rdd_df = rdd.toDF(["title","Token","TF","IDF","TF-IDF"])

# COMMAND ----------

tfidf_rdd_df.show()

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number
windowDept = Window.partitionBy("title").orderBy(col("TF-IDF").desc())
ranked_tfidf=tfidf_rdd_df.withColumn("rank",row_number().over(windowDept))
ranked_tfidf.filter(col('rank').isin([1,2,3,4,5])).show(truncate=False)

# COMMAND ----------

# save the results to csv
top5_tfidfpd = ranked_tfidf.filter(col('rank').isin([1,2,3,4,5])).toPandas()
fpath = os.path.join(CSV_DIR, "top5_tfidfpd.csv")
top5_tfidfpd.to_csv(fpath)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Add NY Giants at Washington Commanders label back to filtered TF-IDF dataframe 

# COMMAND ----------

giants_at_commanders_tfidf = ranked_tfidf.filter(col('title')=='Post Game Thread: New York Giants at Washington Football Team')

# COMMAND ----------

giants_at_commanders_tfidf.show(5)

# COMMAND ----------


