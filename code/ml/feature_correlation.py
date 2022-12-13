# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is to calculate the correlation among numeric features selected in Cleaning&Prepping_Data for machine learning analysis. 

# COMMAND ----------

from pyspark.sql.functions import *
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import seaborn as sns 
import matplotlib.pyplot as plt


# COMMAND ----------

yankees_df = spark.read.parquet("/FileStore/yankees/yankees_df_ml.parquet")

# COMMAND ----------

yankees_df.printSchema()

# COMMAND ----------

yankees_df.dtypes

# COMMAND ----------

from pyspark.sql.types import *
## get numeric variables
numeric_variables = [f.name for f in yankees_df.schema.fields if isinstance(f.dataType, (IntegerType,	LongType, FloatType,DoubleType))]


# COMMAND ----------

## build correlation matrix

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=numeric_variables, outputCol=vector_col)
df_vector = assembler.transform(yankees_df).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 

corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = numeric_variables, index=numeric_variables) 
corr_matrix_df.style.background_gradient(cmap='coolwarm').set_precision(2)



# COMMAND ----------

import os
PLOT_DIR = os.path.join("../../data", "plots")
CSV_DIR = os.path.join("../../data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# COMMAND ----------

plt.figure(figsize=(16,5))  
sns.heatmap(corr_matrix_df, 
            xticklabels=corr_matrix_df.columns.values,
            yticklabels=corr_matrix_df.columns.values,  cmap="Greens", annot=True,
           )
plt.xticks(rotation=45) 

## Save the plot in the plot dir so that it can be checked in into the repo
plot_fpath = os.path.join(PLOT_DIR, 'ml_feature_corr.png')
plt.savefig(plot_fpath)

plt.show()


# COMMAND ----------

# convert to vector column first
vector_colp = "corr_features"
assemblerp = VectorAssembler(inputCols=numeric_variables, outputCol=vector_colp)
df_vectorp = assemblerp.transform(yankees_df).select(vector_colp)

# get correlation matrix
matrixp = Correlation.corr(df_vectorp, vector_colp, method = 'pearson').collect()[0][0] 
corr_matrixp = matrixp.toArray().tolist() 

corr_matrix_dfp = pd.DataFrame(data=corr_matrix, columns = numeric_variables, index=numeric_variables) 
corr_matrix_dfp.style.background_gradient(cmap='coolwarm').set_precision(2)
