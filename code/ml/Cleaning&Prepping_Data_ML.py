# Databricks notebook source
# MAGIC %md
# MAGIC ##Data Cleaning and Preparation

# COMMAND ----------

from pyspark.sql.functions import *
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Before beginning our machine learning analysis, we must clean and prepare our initial dataset for use. <br>
# MAGIC First let's read in our dataset: <br>

# COMMAND ----------

from pyspark.sql.functions import *
from sparknlp.pretrained import PretrainedPipeline
yankees_df = spark.read.parquet("/FileStore/yankees_with_external/yankees_with_external.parquet")

# COMMAND ----------

yankees_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's alter some of our variables and create new features to better prepare our dataset for our analysis. <br>
# MAGIC Comments explainig each operation are provided in the code snippet below.

# COMMAND ----------

#aggregate walk-offs into the win/losses column
yankees_df = yankees_df.withColumn('W/L', when( (col('W/L') == 'W') | (col('W/L') == 'W-wo'), 'W').otherwise('L') )

#calculate comment length based on the number of characters in the comment
yankees_df = yankees_df.withColumn('comment_length', length(col('body')))

#create a new variable called 'game_location' to signify whether a given game was played at home or away
yankees_df = yankees_df.withColumn('game_location', when( col('Unnamed: 4') == '@', 'home').otherwise('away'))

#cast the length of yankees game into a float so it is more usable
yankees_df = yankees_df.withColumn('game_length', regexp_replace(col('Time'),':','.'))
yankees_df = yankees_df.withColumn('game_length', col('game_length').cast('float'))

# COMMAND ----------

# MAGIC %md
# MAGIC We'll make some minor changes to our sentiment model utlized in the NLP section to now only retrieve the prediction result for the sentiment of a given comment

# COMMAND ----------

@udf
def get_sentiment_data(row):
    return row[0]['metadata']

def get_sentiment(df, text_col='body', merge_results=False):
    '''
    Use pretrained twitter sentiment model to return the model's prediction of whether a comment is positive or negative
    '''
    nlp_pipeline = PretrainedPipeline('analyze_sentimentdl_use_twitter', lang='en')

    result = nlp_pipeline.transform(df.select(col(text_col).alias('text')))
    result = result.withColumn('sentiment_result', explode('sentiment.result'))
    result = result.select('sentiment_result')

    if merge_results:
        df = df.withColumn("join_id", monotonically_increasing_id())
        result = result.withColumn("join_id", monotonically_increasing_id())
        result = df.join(result, 'join_id', 'inner')
        result = result.drop('join_id')
        
    return result

# COMMAND ----------

#retrieve the predicted sentiment of each comment
yankees_df = get_sentiment(yankees_df, merge_results=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Our dataset contains all of the comments from 154 Yankees game threads for the Yankees' 2021 season, along with external data about each game and data about each submission post. <br>
# MAGIC Because our dataset contains a significant amount of variables, we will filter down our dataset to only contain variables which we think may be relevant to our machine learning analysis. <br>
# MAGIC We have identified the following variables as relevant: 
# MAGIC <li>num_comments - the number of comments in the game thread (first business question)
# MAGIC <li>W/L - the outcome of the Yankees game for the game/game thread (second business question)
# MAGIC <li>comment_length - the number of characters in a given comment</li>
# MAGIC <li>gilded_cm - the number of time a comment received Reddit gold </li>
# MAGIC <li>controversiality - Number that indicates whether the comment is controversial</li>
# MAGIC <li>sentiment_result - the sentiment of a comment as determined by the twitter sentiment analysis model </li>
# MAGIC <li>score_cm - The score of the comment. The score is the number of upvotes minus the number of downvotes. </li>
# MAGIC <li>score- The score that the submission has accumulated. The score is the number of upvotes minus the number of downvotes. </li>
# MAGIC <li>Opp - the team the Yankees played against for a given game thread </li>
# MAGIC <li>R - the number of runs the Yankees scored for that game/game thread</li>
# MAGIC <li>game_length - the length of the Yankees game (in hours)</li>
# MAGIC <li>D/N - denotes whether the Yankees game was played during the day or during the night</li>
# MAGIC <li>attendance - the number of tickets sold for the game</li>
# MAGIC <li>cLI - statistic that denotes that the importance of the game's outcome on the chances of the Yankees winning a world series </li>
# MAGIC <li>game_location - denotes whether the Yankees game was played at home or away

# COMMAND ----------

keep_cols = ['num_comments', 'W/L','comment_length', 'gilded_cm', 'controversiality', 'sentiment_result', 'score_cm', 'score', 'Opp', 'R', 'game_length', 'D/N', 'attendance', 'cLI', 'game_location']

# COMMAND ----------

test_df = yankees_df

# COMMAND ----------

yankees_df_final = yankees_df.select(keep_cols)

# COMMAND ----------

yankees_df_final.printSchema()

# COMMAND ----------

yankees_df_final.show(3, truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's save our edited dataset to a parquet!

# COMMAND ----------

yankees_df_final.write.parquet("/FileStore/yankees/yankees_df_ml.parquet")
