from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext("local[*]")
spark = SparkSession.builder.getOrCreate()

from pyspark.sql.functions import when
import numpy as np
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import ChiSqSelector

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

tFile="/Users/miguelmitra/Documents/BU/Fall 2024/CS 777/Project/airline_passenger_satisfaction.csv"
df = spark.read.csv(tFile,header=True)

# 1 = satistied, 0 = neutral or dissatisfied
df = df.withColumn("Satisfaction", when(df["Satisfaction"] == "Satisfied", 1).otherwise(0))

# model out of numberical variables of the services the airlines provide
feature_columns = ['Departure and Arrival Time Convenience', 'Ease of Online Booking',
       'Check-in Service', 'Online Boarding', 'Gate Location',
       'On-board Service', 'Seat Comfort', 'Leg Room Service', 'Cleanliness',
       'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
       'In-flight Entertainment', 'Baggage Handling']


def create_pipeline(feature_columns, df):
    stages = []

    # Convert each categorical column to a numerical column using StringIndexer
    for col in feature_columns:
        indexer = StringIndexer(inputCol=col, outputCol=col + "_indexed")
        stages.append(indexer)

    # After indexing, use the new '_indexed' columns as features
    indexed_feature_columns = [col + "_indexed" for col in feature_columns]

    # Now create the VectorAssembler using the indexed feature columns
    assembler = VectorAssembler(inputCols=indexed_feature_columns, outputCol="features")
    stages.append(assembler)

    # Create the pipeline
    pipeline = Pipeline(stages=stages)

    # Fit the model and transform
    model = pipeline.fit(df)
    data = model.transform(df)
    
    return data, indexed_feature_columns

def m_metrics_l(ml_model, test_data):
    # Get predictions and labels
    predictions = ml_model.transform(test_data)
    predictionAndLabels = predictions.select("Satisfaction", "prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()

    # Use BinaryClassificationMetrics for binary classification
    binary_metrics = BinaryClassificationMetrics(predictionAndLabels)

    # AUC and other binary metrics
    auc = binary_metrics.areaUnderROC
    print("Area Under ROC:", auc)

    # Use MulticlassMetrics for confusion matrix, precision, etc.
    multiclass_metrics = MulticlassMetrics(predictionAndLabels)
    precision = multiclass_metrics.precision(1.0)
    recall = multiclass_metrics.recall(1.0)
    f1Score = multiclass_metrics.fMeasure(1.0)
    accuracy = multiclass_metrics.accuracy

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1Score)
    print("Confusion Matrix:\n", multiclass_metrics.confusionMatrix().toArray().astype(int))
    
    
def get_top_features(lr_model,indexed_feature_columns, top_n=5):
    # Get the coefficients and feature names
    coefficients = lr_model.coefficients.toArray()
    feature_names = indexed_feature_columns  # These are the names of your features
    
    # Zip coefficients with feature names and sort them by absolute value
    feature_importance = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
    
    # Print the top 5 most important features
    top_5_features = feature_importance[:5]
    top_5 = []
    
    print("Top 5 features for Logistic Regression:")
    for feature, coef in top_5_features:
        top_5.append(feature)
        print(f"{feature}: {coef}")
        
    return top_5
        
def svc_top_features(sv_model):
    coefficients = sv_model.coefficients
    intercept = sv_model.intercept

    # Get feature names from the indexed feature columns
    feature_names = indexed_feature_columns  # These should match the features used for training

    # Combine feature names and coefficients into a list of tuples
    feature_importance = [(feature, coef) for feature, coef in zip(feature_names, coefficients)]

    # Sort features by the absolute value of their coefficients in descending order
    feature_importance_sorted = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    top_5 = []
    
    # Display sorted coefficients with feature names
    print("\nSVM Coefficients ordered by Importance:")
    for feature, coef in feature_importance_sorted:
        top_5.append(feature)
        print(f"{feature}: {coef}")
        
    return top_5

def map_back_to_original_features(indexed_features):
    original_features = [feature.replace("_indexed", "") for feature in indexed_features]
    return original_features
        
        
data, indexed_feature_columns = create_pipeline(feature_columns, df)
train_data, test_data = data.randomSplit([0.7, 0.3], seed=10)

# Logistic Regression
classifier = LogisticRegression(labelCol="Satisfaction", featuresCol="features", maxIter=100, regParam=0.00001)
# Train model
pipeline = Pipeline(stages=[classifier])
lr_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nLogistic Regression:")
m_metrics_l(lr_model, test_data)

# SVC
classifier = LinearSVC(labelCol="Satisfaction", featuresCol="features",maxIter=100, regParam=0.0001) 
# Train model
pipeline = Pipeline(stages=[classifier])
sv_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nSVM:")
m_metrics_l(sv_model, test_data)

top_5_lr =[]
top_5_svc = []
# extract top 5 features using Logistic regression
lr_model = lr_model.stages[-1]  # The LogisticRegression stage of the pipeline
top_5_lr = get_top_features(lr_model, indexed_feature_columns, top_n=5)

sv_model = sv_model.stages[-1]  # extract top 5 features using svc
top_5_svc = svc_top_features(sv_model)


# Perform analysis based only on top 5 features:
# Map back the top features to their original names (remove '_indexed' suffix)
original_top_5 = map_back_to_original_features(top_5_lr)

# Create a new pipeline using the original top 5 features
data, indexed_top_5_svc_columns = create_pipeline(original_top_5, df)

# Split data again
train_data, test_data = data.randomSplit([0.7, 0.3], seed=10)

# Logistic Regression with top 5 features from SVC
classifier = LogisticRegression(labelCol="Satisfaction", featuresCol="features", maxIter=100, regParam=0.00001)
pipeline = Pipeline(stages=[classifier])
lr_model = pipeline.fit(train_data)

# Evaluate the model
print("\nLogistic Regression (with top 5 features):")
m_metrics_l(lr_model, test_data)


# SVC
classifier = LinearSVC(labelCol="Satisfaction", featuresCol="features",maxIter=100, regParam=0.0001) 
pipeline = Pipeline(stages=[classifier])
sv_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nSVM (with top 5 features):")
m_metrics_l(sv_model, test_data)


####################### Business Class #######################
business_class_df = df.filter(df["Class"] == "Business")
data, indexed_feature_columns = create_pipeline(feature_columns, business_class_df)
# Update the train/test split with scaled data
train_data, test_data = data.randomSplit([0.7, 0.3], seed=10)

# Logistic Regression
classifier = LogisticRegression(labelCol="Satisfaction", featuresCol="features", maxIter=100, regParam=0.00001)
# Train model
pipeline = Pipeline(stages=[classifier])
lr_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nLogistic Regression Business Class:")
m_metrics_l(lr_model, test_data)

# SVC
classifier = LinearSVC(labelCol="Satisfaction", featuresCol="features",maxIter=100, regParam=0.0001) 
# Train model
pipeline = Pipeline(stages=[classifier])
sv_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nSVM Business Class:")
m_metrics_l(sv_model, test_data)

lr_model = lr_model.stages[-1]  # The LogisticRegression stage of the pipeline
get_top_features(lr_model, indexed_feature_columns, top_n=5)

sv_model = sv_model.stages[-1]  # USE SVC because it had the higher accuracy score
svc_top_features(sv_model)


####################### Economy Class #######################
eco_class_df = df.filter(df["Class"] != "Business")
data, indexed_feature_columns = create_pipeline(feature_columns, eco_class_df)
# Update the train/test split with scaled data
train_data, test_data = data.randomSplit([0.7, 0.3], seed=10)

# Logistic Regression
classifier = LogisticRegression(labelCol="Satisfaction", featuresCol="features", maxIter=100, regParam=0.00001)
# Train model
pipeline = Pipeline(stages=[classifier])
lr_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nLogistic Regression Economy Class:")
m_metrics_l(lr_model, test_data)

# SVC
classifier = LinearSVC(labelCol="Satisfaction", featuresCol="features",maxIter=100, regParam=0.0001) 
# Train model
pipeline = Pipeline(stages=[classifier])
sv_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nSVM Economy Class:")
m_metrics_l(sv_model, test_data)

lr_model = lr_model.stages[-1]  # The LogisticRegression stage of the pipeline
get_top_features(lr_model, indexed_feature_columns, top_n=5)

sv_model = sv_model.stages[-1]  # extract top 5 features using svc
svc_top_features(sv_model)


####################### Short Haul Flights #######################
short_haul_df = df.filter(df["Flight Distance"] <= 3000)
data, indexed_feature_columns = create_pipeline(feature_columns, short_haul_df)
# Update the train/test split with scaled data
train_data, test_data = data.randomSplit([0.7, 0.3], seed=10)

# Logistic Regression
classifier = LogisticRegression(labelCol="Satisfaction", featuresCol="features", maxIter=100, regParam=0.00001)
# Train model
pipeline = Pipeline(stages=[classifier])
lr_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nLogistic Regression Short Haul Flight:")
m_metrics_l(lr_model, test_data)

# SVC
classifier = LinearSVC(labelCol="Satisfaction", featuresCol="features",maxIter=100, regParam=0.0001) 
# Train model
pipeline = Pipeline(stages=[classifier])
sv_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nSVM Short Haul Flight:")
m_metrics_l(sv_model, test_data)

lr_model = lr_model.stages[-1]  # The LogisticRegression stage of the pipeline
get_top_features(lr_model, indexed_feature_columns, top_n=5)

sv_model = sv_model.stages[-1]  # extract top 5 features using svc
svc_top_features(sv_model)


####################### Short Haul & Business Class #######################
short_haul_Business_df = df.filter((col("Flight Distance") <= 3000) & (col("Class") == "Business"))
data, indexed_feature_columns = create_pipeline(feature_columns, short_haul_Business_df)
# Update the train/test split with scaled data
train_data, test_data = data.randomSplit([0.7, 0.3], seed=10)

# Logistic Regression
classifier = LogisticRegression(labelCol="Satisfaction", featuresCol="features", maxIter=100, regParam=0.00001)
# Train model
pipeline = Pipeline(stages=[classifier])
lr_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nLogistic Regression Short Haul Flight (Business):")
m_metrics_l(lr_model, test_data)

# SVC
classifier = LinearSVC(labelCol="Satisfaction", featuresCol="features",maxIter=100, regParam=0.0001) 
# Train model
pipeline = Pipeline(stages=[classifier])
sv_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nSVM Short Haul Flight (Business):")
m_metrics_l(sv_model, test_data)

lr_model = lr_model.stages[-1]  # The LogisticRegression stage of the pipeline
get_top_features(lr_model, indexed_feature_columns, top_n=5)

sv_model = sv_model.stages[-1]  # extract top 5 features using svc
svc_top_features(sv_model)

####################### Short Haul & Economy Class #######################
short_haul_Economy_df = df.filter((col("Flight Distance") <= 3000) & (col("Class") != "Business"))
data, indexed_feature_columns = create_pipeline(feature_columns, short_haul_Economy_df)
# Update the train/test split with scaled data
train_data, test_data = data.randomSplit([0.7, 0.3], seed=10)

# Logistic Regression
classifier = LogisticRegression(labelCol="Satisfaction", featuresCol="features", maxIter=100, regParam=0.00001)
# Train model
pipeline = Pipeline(stages=[classifier])
lr_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nLogistic Regression Short Haul Flight (Economy):")
m_metrics_l(lr_model, test_data)

# SVC
classifier = LinearSVC(labelCol="Satisfaction", featuresCol="features",maxIter=100, regParam=0.0001) 
# Train model
pipeline = Pipeline(stages=[classifier])
sv_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nSVM Short Haul Flight (Economy):")
m_metrics_l(sv_model, test_data)

lr_model = lr_model.stages[-1]  # The LogisticRegression stage of the pipeline
get_top_features(lr_model, indexed_feature_columns, top_n=5)

sv_model = sv_model.stages[-1]  # extract top 5 features using svc
svc_top_features(sv_model)



####################### Long Haul Flights #######################
long_haul_df = df.filter(df["Flight Distance"] > 3000)
data, indexed_feature_columns = create_pipeline(feature_columns, long_haul_df)
# Update the train/test split with scaled data
train_data, test_data = data.randomSplit([0.7, 0.3], seed=10)

# Logistic Regression
classifier = LogisticRegression(labelCol="Satisfaction", featuresCol="features", maxIter=100, regParam=0.00001)
# Train model
pipeline = Pipeline(stages=[classifier])
lr_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nLogistic Regression Long Haul Flight:")
m_metrics_l(lr_model, test_data)

# SVC
classifier = LinearSVC(labelCol="Satisfaction", featuresCol="features",maxIter=100, regParam=0.0001) 
# Train model
pipeline = Pipeline(stages=[classifier])
sv_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nSVM Long Haul Flight:")
m_metrics_l(sv_model, test_data)

lr_model = lr_model.stages[-1]  # The LogisticRegression stage of the pipeline
get_top_features(lr_model, indexed_feature_columns, top_n=5)

sv_model = sv_model.stages[-1]  # extract top 5 features using svc
svc_top_features(sv_model)


####################### Long Haul & Business Class #######################
long_haul_Business_df = df.filter((col("Flight Distance") > 3000) & (col("Class") == "Business"))
data, indexed_feature_columns = create_pipeline(feature_columns, long_haul_Business_df)
# Update the train/test split with scaled data
train_data, test_data = data.randomSplit([0.7, 0.3], seed=10)

# Logistic Regression
classifier = LogisticRegression(labelCol="Satisfaction", featuresCol="features", maxIter=100, regParam=0.00001)
# Train model
pipeline = Pipeline(stages=[classifier])
lr_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nLogistic Regression Long Haul Flight (Business):")

# SVC
classifier = LinearSVC(labelCol="Satisfaction", featuresCol="features",maxIter=100, regParam=0.0001) 
# Train model
pipeline = Pipeline(stages=[classifier])
sv_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nSVM Long Haul Flight (Business):")
m_metrics_l(sv_model, test_data)

lr_model = lr_model.stages[-1]  # The LogisticRegression stage of the pipeline
get_top_features(lr_model, indexed_feature_columns, top_n=5)

sv_model = sv_model.stages[-1]  # extract top 5 features using svc
svc_top_features(sv_model)


####################### Long Haul & Economy Class #######################
long_haul_Economy_df = df.filter((col("Flight Distance") > 3000) & (col("Class")!= "Business"))
data, indexed_feature_columns = create_pipeline(feature_columns, long_haul_Economy_df)
# Update the train/test split with scaled data
train_data, test_data = data.randomSplit([0.7, 0.3], seed=10)

# Logistic Regression
classifier = LogisticRegression(labelCol="Satisfaction", featuresCol="features", maxIter=100, regParam=0.00001)
# Train model
pipeline = Pipeline(stages=[classifier])
lr_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nLogistic Regression Long Haul Flight (Economy):")
m_metrics_l(lr_model, test_data)

# SVC
classifier = LinearSVC(labelCol="Satisfaction", featuresCol="features",maxIter=100, regParam=0.0001) 
# Train model
pipeline = Pipeline(stages=[classifier])
sv_model = pipeline.fit(train_data)
# Evaluate the model and get the evaluation time
print("\nSVMLong Haul Flight (Economy):")
m_metrics_l(sv_model, test_data)

lr_model = lr_model.stages[-1]  # The LogisticRegression stage of the pipeline
get_top_features(lr_model, indexed_feature_columns, top_n=5)

sv_model = sv_model.stages[-1]  # extract top 5 features using svc
svc_top_features(sv_model)



sc.stop()




