# Databricks notebook source
# MAGIC %md
# MAGIC #Sentiment Analysis EDA

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we perform sentiment analysis on the Yankees subreddit, and see if and how we can correlate sentiment to <br>
# MAGIC real performance. We find some modest but measurable correlation in sentiment and result (win/loss). It may be interesting <br>
# MAGIC in future work to see how accurate of a classifier we can make to predict result from sentiment, and other metadata related <br>
# MAGIC to the Reddit submissions/comments.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Import Libraries and Helper Functions

# COMMAND ----------

import pandas as pd
import numpy as np
import json
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def BOLD(string):
    return '\033[1m' + string + '\033[0m'

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load Joined Yankees and Reddit Data

# COMMAND ----------

mlb_df = spark.read.parquet("/FileStore/yankees_with_external/yankees_with_external.parquet").cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ###View Data

# COMMAND ----------

mlb_df.limit(5).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Analyze Yankees Subreddit

# COMMAND ----------

print(f'Number of posts/comments in yankees subreddit within time period:    {mlb_df.count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sentiment Analysis Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define Pipeline

# COMMAND ----------

@udf
def get_sentiment_data(row):
    return row[0]['metadata']

@udf
def get_pos_prob(row):
    return row['positive']

@udf
def get_neg_prob(row):
    return row['negative']


def get_sentiment(df, text_col='body', merge_results=False):
    '''
    Use pretrained twitter sentiment model to return probability that post/comment is positive/negative.
    '''
    nlp_pipeline = PretrainedPipeline('analyze_sentimentdl_use_twitter', lang='en')

    result = nlp_pipeline.transform(df.select(F.col(text_col).alias('text')))
    result = result.withColumn('extracted_sentiment', get_sentiment_data(result.sentiment))
    result = result.withColumn('negative_prob', get_neg_prob(result.extracted_sentiment).cast('float'))
    result = result.withColumn('positive_prob', get_pos_prob(result.extracted_sentiment).cast('float'))
    
    
    result = result.select(['negative_prob', 'positive_prob'])

    if merge_results:
        df = df.withColumn("join_id", F.monotonically_increasing_id())
        result = result.withColumn("join_id", F.monotonically_increasing_id())
        result = df.join(result, 'join_id', 'inner')
        result = result.drop('join_id')
        
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC We use a pretrained sentiment model trained on Twitter data from John Snow Labs. This will read the <br>
# MAGIC body of the reddit comments/submissions and return predicted probabilites that the sentiment is <br>
# MAGIC negative, or positive.

# COMMAND ----------

# MAGIC %md
# MAGIC ####Perform Sentiment Analysis

# COMMAND ----------

sentiment = get_sentiment(mlb_df, merge_results=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ###View Positive/Negative Sentiment By Result
# MAGIC Let's group the submissions/comments by result (win/loss) and look at the average sentiment.

# COMMAND ----------

keep_cols = ['Gm#', 'W/L', 'positive_prob', 'negative_prob']

sub_mlb = sentiment.select(keep_cols)

avg_sentiment_by_result = sub_mlb.groupBy('W/L').avg('positive_prob', 'negative_prob').toPandas().set_index('W/L')
avg_sentiment_by_result.columns = ['positive_sentiment_probability_avg', 'negative_sentiment_probability_avg']
avg_sentiment_by_result

# COMMAND ----------

plt.style.use('fivethirtyeight')
avg_sentiment_by_result.plot.bar(title='Average Sentiment By Result', fontsize=12)
plt.xlabel('Result', fontsize=14)
plt.ylabel('Sentiment', fontsize=14)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's filter down to get only wins and losses to get a better view. <br>

# COMMAND ----------

## Filter out non wins / losses
_sub_mlb = sub_mlb.filter(F.col('W/L').isin(['W', 'L']))

_avg_sentiment_by_result = _sub_mlb.groupBy('W/L').avg('positive_prob', 'negative_prob').toPandas().set_index('W/L')

_avg_sentiment_by_result.columns = ['positive_sentiment_probability_avg', 'negative_sentiment_probability_avg']

# COMMAND ----------

_avg_sentiment_by_result.plot.bar(title='Average Sentiment By Result', fontsize=12)
plt.style.use('fivethirtyeight')
plt.xlabel('Result', fontsize=14)
plt.ylabel('Sentiment', fontsize=14)
plt.show()

plt.savefig('sentiment_analysis_fig_1a.png')

# COMMAND ----------

# MAGIC %md
# MAGIC Sentiment of all posts seems to be generally negative, however it does also seem that sentiment is more positive when the Yankees won. <br>
# MAGIC Let's see if the Pearson correlation supports this hypothesis.

# COMMAND ----------

@udf
def cast_to_int(val):
    return int(val == 'W')

_sub_mlb = _sub_mlb.withColumn('W/L', cast_to_int(_sub_mlb['W/L']).cast('int'))

corr = _sub_mlb.corr('positive_prob', 'W/L', method='pearson')

print(BOLD(f'Pearson correlation between Yankees W/L and sentiment in subreddit:   {corr}'))

# COMMAND ----------

# MAGIC %md
# MAGIC Interesting. It seems that the result (win/loss) is weakly correlated to the sentiment of the post. <br>
# MAGIC In the future, we may attempt to create a classifier which takes in as input, the sentiment of all of the tweets related to a game and <br>
# MAGIC some non-information leaking metadata and tries to predict the result of the game.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Player Sentiment
# MAGIC Let's now switch things up and look at the sentiment towards different players. Specifically `Aaron Judge` and `Gary Sanchez`. <br>

# COMMAND ----------

# MAGIC %md
# MAGIC First use regex to create dummy variables `has_aaron_judge` and `has_gary_sanchez`.

# COMMAND ----------

## Start with the sentiment dataframe from before, which has
## all original data and sentiment cols.
from pyspark.sql.functions import *

player_sentiment = sentiment.withColumn(
    'has_aaron_judge',
    F.col('body').rlike('(?i)Aaron Judge|(?i)Judge|(?i)All Rise')
)

player_sentiment = player_sentiment.withColumn(
    'has_gary_sanchez',
    F.col('body').rlike('(?i)Gary Sanchez|(?i)Sanchez|(?i)Kraken')
)

# COMMAND ----------

keep_cols = ['Gm#', 'W/L', 'positive_prob', 'negative_prob', 'has_aaron_judge', 'has_gary_sanchez']

player_sentiment = player_sentiment.select(keep_cols)

player_sentiment.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's plot the sentiment of comments towards these players and measure the correlations.

# COMMAND ----------

avg_sentiment_by_judge = player_sentiment.groupBy('has_aaron_judge').avg('positive_prob', 'negative_prob').toPandas()
avg_sentiment_by_judge = avg_sentiment_by_judge.set_index('has_aaron_judge')
avg_sentiment_by_judge.columns = ['judge_positive_sentiment_probability_avg', 'judge_negative_sentiment_probability_avg']

avg_sentiment_by_sanchez = player_sentiment.groupBy('has_gary_sanchez').avg('positive_prob', 'negative_prob').toPandas()
avg_sentiment_by_sanchez = avg_sentiment_by_sanchez.set_index('has_gary_sanchez')
avg_sentiment_by_sanchez.columns = ['sanchez_positive_sentiment_probability_avg', 'sanchez_negative_sentiment_probability_avg']

avg_sentiment_by_player = pd.concat([avg_sentiment_by_sanchez, avg_sentiment_by_judge], axis=1)
avg_sentiment_by_player

# COMMAND ----------

_player_sentiment = player_sentiment.withColumn('has_aaron_judge', F.col('has_aaron_judge').cast('int'))
_player_sentiment = _player_sentiment.withColumn('has_gary_sanchez', F.col('has_gary_sanchez').cast('int'))

corr_judge   = _player_sentiment.corr('positive_prob', 'has_aaron_judge', method='pearson')
corr_sanchez = _player_sentiment.corr('positive_prob', 'has_gary_sanchez', method='pearson')

print(BOLD(f'Pearson correlation between mention of Aaron Judge and sentiment in subreddit:   {corr_judge}'))
print(BOLD(f'Pearson correlation between mention of Gary Sanchez and sentiment in subreddit:   {corr_sanchez}'))

# COMMAND ----------

# MAGIC %md
# MAGIC Interesting, it looks like posts with Gary Sanchez are on average more negative and posts with Aaron Judge <br>
# MAGIC are more positive (as compared to the average post (no mention of them)). It seems Aaron Sanchez is not very <br>
# MAGIC well liked in the Yankees subreddit. <br>
# MAGIC 
# MAGIC It should be noted however that these correlations are fairly weak. Not definitive.
