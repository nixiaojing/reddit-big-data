# Databricks notebook source
import pandas as pd
from pyspark.sql.functions import *
from pyspark.ml import PipelineModel
from pyspark.ml import Pipeline, Model
from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler,StandardScaler
from pyspark.ml.classification import LinearSVC,OneVsRest
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


# COMMAND ----------

# MAGIC %md
# MAGIC ## Read data for machine learning analysis <br>
# MAGIC processed in Cleaning&Prepping_Data notebook

# COMMAND ----------

yankees_df = spark.read.parquet("/FileStore/yankees/yankees_df_ml.parquet")

# COMMAND ----------

## delete na rows
yankees_df = yankees_df.dropna()


# COMMAND ----------

yankees_df.printSchema()


# COMMAND ----------

yankees_df.count()

# COMMAND ----------

#yankees_df = yankees_df.filter(yankees_df.sentiment_result != 'neutral')

# COMMAND ----------

## sample a small dataset
small_yankees_df = yankees_df.sample(withReplacement=True, fraction=0.01, seed=3)
small_yankees_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data processing for Support Vector Machine and Random Forest
# MAGIC <!-- We are going to apply support vector machine (SVM) on the data to predict game results (win/loss). SVM requires numeric features. Thus, we are going to convert our categorical variables into dummy variables. Those variables include sentiment_result, Opp, D_N, game_location. -->
# MAGIC Since SVM and Random Forest do not work well on imbalanced data, we need first check the balance of the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check imbalance data

# COMMAND ----------

yankees_df.groupby('W/L').count().collect()

# COMMAND ----------

yankees_df.groupby('sentiment_result').count().collect()

# COMMAND ----------

yankees_df.groupby('Opp').count().collect()

# COMMAND ----------

yankees_df.groupby('game_location').count().collect()

# COMMAND ----------

# MAGIC %md
# MAGIC Dealing with imbalanced data <br>
# MAGIC https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/<br>
# MAGIC Example:
# MAGIC "
# MAGIC For  heart stroke example:
# MAGIC 
# MAGIC n_samples=  43400,  n_classes= 2(0&1), n_sample0= 42617, n_samples1= 783
# MAGIC 
# MAGIC Weights for class 0:
# MAGIC 
# MAGIC w0=  43400/(2*42617) = 0.509
# MAGIC 
# MAGIC Weights for class 1:
# MAGIC 
# MAGIC w1= 43400/(2*783) = 27.713 "
# MAGIC <br><br><br>
# MAGIC 
# MAGIC 
# MAGIC Our case:
# MAGIC [Row(sentiment_result='positive', count=221101),
# MAGIC  Row(sentiment_result='neutral', count=31770),
# MAGIC  Row(sentiment_result='negative', count=366462)]
# MAGIC 
# MAGIC n_samples=  6201,  n_classes= 3(positive, negative, neutral), n_sample(positive)= 221101, n_samples(neutral)= 31770,n_samples(neutral)= 366462
# MAGIC 
# MAGIC Weights for class positive:
# MAGIC 
# MAGIC w0=  619333/(3*221101) = 0.934
# MAGIC 
# MAGIC Weights for class neutral:
# MAGIC 
# MAGIC w1= 619333/(3*31770) = 6.50
# MAGIC 
# MAGIC 
# MAGIC Weights for class negative:
# MAGIC 
# MAGIC w2= 619333/(3*366462) = 0.56

# COMMAND ----------

## add a weight col to handle imbalance
yankees_df = yankees_df.withColumn('weight', when(col('sentiment_result') == 'positive', 0.934)\
                                               .when(col('sentiment_result') == 'neutral',  6.50)
    .otherwise(0.56) ) 
yankees_df.select('sentiment_result','weight').show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build pipeline for Random Forest model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split data into train, test, and split

# COMMAND ----------

train_data, test_data, predict_data = yankees_df.randomSplit([0.8, 0.18, 0.02], 24)


# COMMAND ----------

print("Number of training records: " + str(train_data.count()))
print("Number of testing records : " + str(test_data.count()))
print("Number of prediction records : " + str(predict_data.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create pipeline and train a model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC ### The ordering behavior is controlled by setting stringOrderType. Its default value is ‘frequencyDesc’

# COMMAND ----------

stringIndexer_WL = StringIndexer(inputCol="W/L", outputCol="WL_ix")
stringIndexer_sentiment= StringIndexer(inputCol="sentiment_result", outputCol="sentiment_ix")
stringIndexer_Opp = StringIndexer(inputCol="Opp", outputCol="Opp_ix")
stringIndexer_DN = StringIndexer(inputCol="D/N", outputCol="DN_ix")
stringIndexer_game_location = StringIndexer(inputCol="game_location", outputCol="game_location_ix")

# COMMAND ----------

onehot_WL = OneHotEncoder(inputCol="WL_ix", outputCol="WL_vec")
onehot_sentiment = OneHotEncoder(inputCol="sentiment_ix", outputCol="sentiment_vec")
onehot_Opp = OneHotEncoder(inputCol="Opp_ix", outputCol="Opp_vec")
onehot_DN = OneHotEncoder(inputCol="DN_ix", outputCol="DN_vec")
onehot_game_location = OneHotEncoder(inputCol="game_location_ix", outputCol="game_location_vec")

# COMMAND ----------

vectorAssembler_features = VectorAssembler(
    inputCols=['comment_length','gilded_cm','controversiality','score_cm', 'score', 'R','game_length','attendance','cLI','WL_vec','Opp_vec','DN_vec','game_location_vec'], 
    outputCol= "features")

# COMMAND ----------

rf = RandomForestClassifier(labelCol="sentiment_ix", featuresCol="features", numTrees=2000, weightCol='weight', impurity='entropy')


# COMMAND ----------

labelConverter = IndexToString(inputCol="prediction", 
                               outputCol="predictedSentiment",
                              labels = ['negative', 'positive', 'neutral'])

# COMMAND ----------

pipeline_rf = Pipeline(stages=[stringIndexer_WL,
                               stringIndexer_Opp,
                               stringIndexer_DN,
                               stringIndexer_game_location,
                               stringIndexer_sentiment,
                               onehot_WL,
                               onehot_DN,
                               onehot_Opp,
                               onehot_game_location,
                               onehot_sentiment,
                               vectorAssembler_features,
                               rf, labelConverter])

# COMMAND ----------

train_data.printSchema()

# COMMAND ----------

train_data.show(10)

# COMMAND ----------

model_rf = pipeline_rf.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Model Test Results

# COMMAND ----------

predictions = model_rf.transform(test_data)

# COMMAND ----------

predictions.show()

# COMMAND ----------

evaluatorRF = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions)

# COMMAND ----------

print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))

# COMMAND ----------

predictions_train = model_rf.transform(train_data)

# COMMAND ----------

evaluatorRF = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions_train)

# COMMAND ----------

print("Accuracy = %g" % accuracy)
print("Train Error = %g" % (1.0 - accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Confusion Matrix

# COMMAND ----------

from sklearn.metrics import confusion_matrix

# COMMAND ----------

predictions.show()

# COMMAND ----------

y_pred=predictions.select("prediction").collect()
y_orig=predictions.select("sentiment_ix").collect()

# COMMAND ----------

cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)

# COMMAND ----------

## confusion matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'fontsize': 20, 'fontweight': 'bold'}, pad=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontdict={'fontsize': 19, 'fontweight': 'bold'})
    plt.xlabel('Predicted label',fontdict={'fontsize': 19, 'fontweight': 'bold'}, labelpad=20)

# COMMAND ----------

class_names = ['negative','positive','neutral']

# COMMAND ----------

from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi=80)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)

plot_confusion_matrix(cm, classes=class_names,title='Confusion Matrix Random Forest',cmap='GnBu')

# COMMAND ----------

# MAGIC %md
# MAGIC ### f1 score

# COMMAND ----------

evaluator_rf = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction")
f1_result_rf = evaluator_rf.evaluate(predictions)
f1_result_rf

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the model

# COMMAND ----------

model_rf.save("/FileStore/MLmodel/")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Build pipeline for SVM with One vs Rest

# COMMAND ----------

numeric_features_svm = VectorAssembler(inputCols=['comment_length','gilded_cm','controversiality','score_cm', 'score', 'R','game_length','attendance','cLI'], 
    outputCol= "numeric_features_svm")

scaler = StandardScaler(inputCol="numeric_features_svm", outputCol="scaledFeatures",
                        withStd=True, withMean=True)

# COMMAND ----------

vectorAssembler_features_svm = VectorAssembler(
    inputCols=['scaledFeatures','WL_vec','Opp_vec','DN_vec','game_location_vec'], 
    outputCol= "features")

# COMMAND ----------

# svm = LinearSVC()
ovr = OneVsRest(labelCol="sentiment_ix", featuresCol="features", weightCol="weight",classifier=LinearSVC())


# COMMAND ----------

# nb=NaiveBayes(labelCol="sentiment_ix", featuresCol="features", weightCol="weight")

# COMMAND ----------

labelConverter_svm = IndexToString(inputCol="prediction", 
                               outputCol="predictedSentiment",
                              labels = ['negative', 'positive', 'neutral'])

# COMMAND ----------

pipeline_svm = Pipeline(stages=[stringIndexer_WL,
                               stringIndexer_Opp,
                               stringIndexer_DN,
                               stringIndexer_game_location,
                               stringIndexer_sentiment,
                               onehot_WL,
                               onehot_DN,
                               onehot_Opp,
                               onehot_game_location,
                               onehot_sentiment,
                               numeric_features_svm,
                               scaler,
                               vectorAssembler_features_svm,
                               ovr,labelConverter_svm])

# COMMAND ----------

model_svm = pipeline_svm.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Model Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC #### test accuracy and error

# COMMAND ----------

predictions_svm = model_svm.transform(test_data)

# COMMAND ----------

evaluator_svm = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction", metricName="accuracy")
accuracy_svm = evaluator_svm.evaluate(predictions_svm)

# COMMAND ----------

print("Accuracy = %g" % accuracy_svm)
print("Test Error = %g" % (1.0 - accuracy_svm))

# COMMAND ----------

# MAGIC %md
# MAGIC #### train accuracy and error

# COMMAND ----------

predictions_train_svm = model_svm.transform(train_data)

# COMMAND ----------

evaluator_svm_train = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction", metricName="accuracy")
accuracy_svm_train = evaluator_svm_train.evaluate(predictions_train_svm)

# COMMAND ----------

print("Accuracy = %g" % accuracy_svm_train)
print("Train Error = %g" % (1.0 - accuracy_svm_train))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Confusion Matrix

# COMMAND ----------

predictions_svm.show()

# COMMAND ----------

y_pred_svm=predictions_svm.select("predictedSentiment").collect()
y_orig_svm=predictions_svm.select("sentiment_result").collect()

# COMMAND ----------

cm_svm = confusion_matrix(y_orig_svm, y_pred_svm)
print("Confusion Matrix:")
print(cm_svm)

# COMMAND ----------

from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi=80)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)

plot_confusion_matrix(cm_svm, classes=class_names,title='Confusion Matrix SVM',cmap='GnBu')

# COMMAND ----------

# MAGIC %md
# MAGIC ### f1 score

# COMMAND ----------

evaluator_svm = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction")
f1_result = evaluator_svm.evaluate(predictions_svm)
f1_result

# COMMAND ----------

# MAGIC %md
# MAGIC ### save the model

# COMMAND ----------

model_svm.save("/FileStore/MLmodel_svm/")

# COMMAND ----------

# MAGIC %md
# MAGIC ##
# MAGIC Comparison
# MAGIC <table>
# MAGIC <tr><td>
# MAGIC 
# MAGIC | |Random Forest|SVM|
# MAGIC |-|-|-|
# MAGIC |Accuracy|53%|36%|
# MAGIC |f1 score|53%|41%|
# MAGIC |Test Error|47%|64%|
# MAGIC 
# MAGIC </td></tr> </table>

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Random Forest model with normalized numerical variables <br>
# MAGIC In this session, we are trying to improve our model with one more transformation of the data

# COMMAND ----------

# MAGIC %md
# MAGIC ### The ordering behavior is controlled by setting stringOrderType. Its default value is ‘frequencyDesc’

# COMMAND ----------

pipeline_rf_norm = Pipeline(stages=[stringIndexer_WL,
                               stringIndexer_Opp,
                               stringIndexer_DN,
                               stringIndexer_game_location,
                               stringIndexer_sentiment,
                               onehot_WL,
                               onehot_DN,
                               onehot_Opp,
                               onehot_game_location,
                               onehot_sentiment,
                               numeric_features_svm,
                               scaler,
                               vectorAssembler_features_svm,
                               rf, labelConverter])

# COMMAND ----------

model_rf_norm = pipeline_rf_norm.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Model Test Results

# COMMAND ----------

predictions_rf_norm = model_rf_norm.transform(test_data)

# COMMAND ----------

evaluatorRF_norm = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction", metricName="accuracy")
accuracy_rf_norm = evaluatorRF_norm.evaluate(predictions_rf_norm)

# COMMAND ----------

print("Accuracy = %g" % accuracy_rf_norm)
print("Test Error = %g" % (1.0 - accuracy_rf_norm))

# COMMAND ----------

predictions_rf_train_norm = model_rf_norm.transform(train_data)

# COMMAND ----------

evaluatorRF_train_norm = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction", metricName="accuracy")
accuracyRF_train_norm = evaluatorRF_train_norm.evaluate(predictions_rf_train_norm)

# COMMAND ----------

print("Accuracy = %g" % accuracyRF_train_norm)
print("Train Error = %g" % (1.0 - accuracyRF_train_norm))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Confusion Matrix

# COMMAND ----------

from sklearn.metrics import confusion_matrix

# COMMAND ----------

y_pred=predictions_rf_norm.select("prediction").collect()
y_orig=predictions_rf_norm.select("sentiment_ix").collect()

# COMMAND ----------

cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)

# COMMAND ----------

## confusion matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'fontsize': 20, 'fontweight': 'bold'}, pad=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontdict={'fontsize': 19, 'fontweight': 'bold'})
    plt.xlabel('Predicted label',fontdict={'fontsize': 19, 'fontweight': 'bold'}, labelpad=20)

# COMMAND ----------

class_names = ['negative','positive','neutral']

# COMMAND ----------

from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi=80)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)

plot_confusion_matrix(cm, classes=class_names,title='Confusion Matrix Random Forest with Normalized features',cmap='GnBu')

# COMMAND ----------

# MAGIC %md
# MAGIC ### f1 score

# COMMAND ----------

evaluator_rf_norm = MulticlassClassificationEvaluator(labelCol="sentiment_ix", predictionCol="prediction")
f1_result_rf_norm = evaluator_rf_norm.evaluate(predictions_rf_norm)
f1_result_rf_norm

# COMMAND ----------

# MAGIC %md
# MAGIC ##
# MAGIC Comparison of the updated models. The results shows that the improvement is minor. 
# MAGIC <table>
# MAGIC <tr><td>
# MAGIC 
# MAGIC | |RF|SVM|RF norm|
# MAGIC |-|-|-|-|
# MAGIC |Accuracy|53%|36%|52%|
# MAGIC |f1 score|53%|41%|53%|
# MAGIC |Test Error|47%|64%|48%|
# MAGIC 
# MAGIC </td></tr> </table >
# MAGIC Note: RF, Random Forest; SVM, support vector machine; RF norm, random forest model with normalized features
