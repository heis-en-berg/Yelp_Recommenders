# Python imports
import os
import copy
import time
import random
from statistics import mean
# pyspark imports
import findspark
findspark.init()
import pyspark
from pyspark import SparkContext
from pyspark.rdd import RDD
# pyspark sql imports
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc, size, max, abs, lit
# pyspark ml imports
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from math import sqrt, log
from operator import add
from pyspark.ml.feature import StringIndexer


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("ALS_RECOMMENDER") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

spark = init_spark()
sc = spark.sparkContext
sc.setCheckpointDir('checkpoint/')
data_path = "E:\Big_data_project\dataset\yelp_dataset"

print('########################## buisiness.json ###########################')
# Reading business.json(135 mb)
business_data = spark.read.json(data_path + '\\business.json')
# Montreal Business and removing businesses with less than 5 ratings
business_mtl_dataframe = business_data.filter(business_data['city'] == 'Montréal')
business_mtl_dataframe = business_mtl_dataframe.filter(business_mtl_dataframe['review_count'] > 5)
# Montreal retaurants
business_mtl_dataframe = business_mtl_dataframe.filter(business_mtl_dataframe.categories.like('%Restaurants%'))
#  String indexer for businesses
indexer = StringIndexer(inputCol="business_id", outputCol="business_index")
business_mtl_dataframe = indexer.fit(business_mtl_dataframe).transform(business_mtl_dataframe)
business_mtl_dataframe = business_mtl_dataframe.withColumn("business_index", business_mtl_dataframe["business_index"].cast(IntegerType()))

# Removinf unecessary columns
restaurants_LV_dataframe = business_mtl_dataframe.drop('address', 'hours', 'is_open', 'latitude', 'longitude', 'postal_code', 'state', 'stars', 'city')

print('########################## review.json ###########################')
reviews_data = spark.read.json(data_path + '\\review.json')
reviews_data = reviews_data.drop('text', 'useful', 'date', 'funny', 'cool', 'review_id')
# get reviews of the restaurants based in Montreal
reviews_data = reviews_data.join(restaurants_LV_dataframe, ['business_id'], 'leftsemi')
reviews_data = reviews_data.join(restaurants_LV_dataframe, ['business_id'])
reviews_data = reviews_data.select('stars', 'user_id', 'business_index')

# indexer = StringIndexer(inputCol="business_id", outputCol="business_index")
# reviews_data = indexer.fit(reviews_data).transform(reviews_data)
# reviews_data = reviews_data.withColumn("business_index", reviews_data["business_index"].cast(IntegerType()))

#  String indexer for ratings
indexer = StringIndexer(inputCol="user_id", outputCol="user_index")
reviews_data = indexer.fit(reviews_data).transform(reviews_data)
reviews_data = reviews_data.withColumn("user_index", reviews_data["user_index"].cast(IntegerType()))

reviews_data = reviews_data.withColumn("stars", reviews_data["stars"].cast(IntegerType()))
# Removing ratings of the user who rated less than 5 users.
review_count_per_user = reviews_data.groupBy('user_index').count()
review_count_per_user = review_count_per_user.rdd.filter(lambda x: x[1] > 4).toDF()
reviews_dataframe = reviews_data.join(review_count_per_user, 'user_index', 'leftsemi')

(training, test) = reviews_dataframe.randomSplit([0.8, 0.2])
print('########################## Training ###########################')
als = ALS(userCol="user_index", itemCol="business_index", ratingCol="stars", coldStartStrategy="drop")
als.setSeed(123)
# Setting parameters for grid builder
grid = ParamGridBuilder().addGrid(als.maxIter, [20]).addGrid(als.rank, [20,30,40,50,60,70]).addGrid(als.regParam, [0.45,0.5,0.55]).build()
evaluator = RegressionEvaluator(predictionCol=als.getPredictionCol(),labelCol=als.getRatingCol(), metricName='rmse')
cv = CrossValidator(estimator=als, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
cvModel = cv.fit(training)

cvModel.save('E:\Big_data_project\model\collab_montreal_model\\bestModel')
predictions = cvModel.transform(test)
predictions.cache()

print('########################## Computing RMSE ###########################')

rmse_evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='stars', metricName='rmse')
mae_evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='stars', metricName='mae')
rmse = rmse_evaluator.evaluate(predictions)
mae = mae_evaluator.evaluate(predictions)

print(rmse)
print(mae)
print('############################### predictions #########################')
# user with amximum rating is indexed to 0
training_bid_with_our_user = reviews_dataframe.where("user_index == 0")
list_bid_our_user = training_bid_with_our_user.select('business_index').collect()
list_bid_our_user_modified = []
for item in list_bid_our_user:
    list_bid_our_user_modified.append(item.business_index)

test_without_user_bid_df = restaurants_LV_dataframe.rdd.filter(lambda x: x.business_index not in list_bid_our_user_modified).toDF(sampleRatio=0.2).select('business_index')
test_without_user_bid_df = test_without_user_bid_df.withColumn('user_index', lit(0))
test_without_user_bid_df.persist()
data_path_model = "E:\Big_data_project\model\collab_montreal_model\\bestModel"
alsModel = ALSModel.load(data_path_model)
predictions_for_user_id_0 = alsModel.transform(test_without_user_bid_df)
predictions_for_user_id_0.cache()
