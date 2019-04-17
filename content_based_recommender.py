    import sys
    import copy
    import time
    import random
    import pyspark
    import pandas as pd
    from statistics import mean
    from pyspark import SparkContext
    from pyspark.rdd import RDD
    from pyspark.sql import Row
    from pyspark.sql import DataFrame
    from pyspark.sql import SparkSession
    from pyspark.ml.fpm import FPGrowth
    from pyspark.sql.functions import desc, size, max, abs, lit, monotonically_increasing_id, countDistinct
    from pyspark.sql.functions import sum as sql_sum
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.sql import Row
    from pyspark.sql import functions as F
    from math import sqrt, log
    from operator import add
    from pyspark.ml.feature import StringIndexer
    from pyspark.sql.types import IntegerType


    MAX_MEMORY = "6g"

    def init_spark():
        spark = SparkSession \
            .builder \
            .config("spark.executor.memory", MAX_MEMORY) \
            .config("spark.driver.memory", MAX_MEMORY) \
            .getOrCreate()
        return spark
    
    spark = init_spark()
    sc = spark.sparkContext
    sc.setCheckpointDir('checkpoint/')
    
    ########################## buisiness.json ###########################
    
    print('########################## buisiness.json ###########################')
    
    business_data = spark.read.json('./data_json/business.json')
    
    # Montréal Businesses
    business_mtl_dataframe = business_data.filter(business_data['city'] == 'Montréal')
    business_mtl_dataframe = business_mtl_dataframe.filter(business_mtl_dataframe['review_count'] > 5)
    
    
    # Montréal Restaurants
    
    business_mtl_dataframe = business_mtl_dataframe.filter(business_mtl_dataframe.categories.like('%Restaurants%'))

    # indexing business id
    indexer = StringIndexer(inputCol="business_id", outputCol="bid")
    business_mtl_dataframe = indexer.fit(business_mtl_dataframe).transform(business_mtl_dataframe)
    business_mtl_dataframe = business_mtl_dataframe.withColumn("bid", business_mtl_dataframe["bid"].cast(IntegerType()))
    
    restaurants_mtl_dataframe = business_mtl_dataframe.drop('address', 'hours', 'is_open', 'latitude', 'longitude', 'postal_code', 'state' , 'stars' , 'city')
    
    ########################## business categories binary df #####################
    
    print('########################## business categories binary df #####################')
    
    def normalize_data(list):
        new_list = []
        for i in range(0,len(list)):
            new_list.append(list[i].lstrip())
        return set(new_list)
    
    categories_RDD = restaurants_mtl_dataframe.rdd.filter(lambda x: x.categories != None)
    categories_RDD = categories_RDD.map(lambda x: x.categories.split(','))
    categories_RDD = categories_RDD.map(normalize_data)
    categories_set = categories_RDD.reduce(lambda x,y: x.union(y))
    categories_set.remove('Restaurants')
    
    def add_categories_as_columns(x, category_set):
        temp_dict = x.asDict()
        restaurant_category = [element.lstrip() for element in x.categories.split(',')]
        for element in category_set:
            temp_dict[element] = 1 if element in restaurant_category else 0
        output = Row(**temp_dict)
        return output

    restaurants_mtl_RDD = restaurants_mtl_dataframe.rdd.filter(lambda x: x.categories != None)
    restaurants_binary_categories_mtl_RDD = restaurants_mtl_RDD.map(lambda x: add_categories_as_columns(x, categories_set))
    
    restaurants_binary_categories_mtl_dataframe = restaurants_binary_categories_mtl_RDD.toDF(sampleRatio=0.2)
    restaurants_binary_categories_mtl_dataframe = restaurants_binary_categories_mtl_dataframe.drop('attributes', 'categories', 'name', 'review_count')

    columns = restaurants_binary_categories_mtl_dataframe.columns
    columns.remove('bid')
    columns = ['bid'] + columns
    restaurants_binary_categories_mtl_dataframe = restaurants_binary_categories_mtl_dataframe.select(columns)
    
     # removing rows with no categories
    restaurants_binary_categories_mtl_dataframe = restaurants_binary_categories_mtl_dataframe.rdd.filter(lambda x : sum(x[1:-1]) != 0).toDF()
    restaurants_binary_categories_mtl_dataframe.persist(pyspark.StorageLevel.DISK_ONLY)


    ################################ DF and IDF of categories #######################
    
    print('################################ DF and IDF of categories #######################')
    
    category_idf_score = {}      
          
    restaurant_count = restaurants_binary_categories_mtl_dataframe.count()
    
    category_idf_score = restaurants_binary_categories_mtl_dataframe.groupBy().sum().collect()[0].asDict()
    
    category_idf_score.pop('sum(bid)')
    
    for category in categories_set:
        
        df_score_category = category_idf_score.pop('sum(' + category + ')')
        idf_score_for_category = log(restaurant_count/df_score_category, 10)
        category_idf_score.update({category: idf_score_for_category})
        
        
        #print("category: " + str(count))
        #category_DF_score = normalized_restaurants_dataframe.select(normalized_restaurants_dataframe[category], F.when(normalized_restaurants_dataframe[category] > 0, 1).otherwise(0).alias('score'))
        #category_DF_score = category_DF_score.groupBy().sum().collect()[0][1]
        #category_idf_score[category] = log(restaurant_count/category_DF_score, 10)

    ########################## review.json ###########################
    
    print('########################## review.json ###########################')
    
    reviews = spark.read.json('./data_json/review.json')
    
    # drop columns
    reviews_data = reviews.drop('text', 'useful', 'date', 'funny', 'cool', 'review_id')

    # index business id and user id
    #indexer = StringIndexer(inputCol="business_id", outputCol="bid")
    #reviews_data = indexer.fit(reviews_data).transform(reviews_data)
    #reviews_data = reviews_data.withColumn("bid", reviews_data["bid"].cast(IntegerType()))
    
     # reviews data from Las Vegas
    reviews_data = reviews_data.join(restaurants_binary_categories_mtl_dataframe, ['business_id'], 'leftsemi')
    reviews_data = reviews_data.join(restaurants_binary_categories_mtl_dataframe, ['business_id'])
    reviews_data = reviews_data.select('stars', 'user_id', 'bid')
    

    restaurants_binary_categories_mtl_dataframe = restaurants_binary_categories_mtl_dataframe.drop('business_id')
    
    indexer = StringIndexer(inputCol="user_id", outputCol="uid")
    reviews_data = indexer.fit(reviews_data).transform(reviews_data)
    reviews_data = reviews_data.withColumn("uid", reviews_data["uid"].cast(IntegerType()))

    # cast star from float to integer
    reviews_data = reviews_data.withColumn("stars", reviews_data["stars"].cast(IntegerType()))

    # drop business_id and user_id
    reviews_dataframe = reviews_data.drop('user_id')
    
    #drop users with less than 5 reviews
    review_count_per_user = reviews_dataframe.groupBy('uid').count()
    review_count_per_user = review_count_per_user.rdd.filter(lambda x: x[1] > 4).toDF()
    reviews_dataframe = reviews_dataframe.join(review_count_per_user, 'uid', 'leftsemi')
    
    # Repeat : drop business with less than 5 reviews
    restaurants_binary_categories_mtl_dataframe = restaurants_binary_categories_mtl_dataframe.join(reviews_dataframe, 'bid', 'leftsemi')
    restaurants_binary_categories_mtl_dataframe.persist(pyspark.StorageLevel.DISK_ONLY)
    
    # Repeat
    #reviews_data = reviews_data.join(restaurants_binary_categories_mtl_dataframe, ['bid'], 'leftsemi')
    
    reviews_dataframe.persist(pyspark.StorageLevel.DISK_ONLY)
    
    ######################## Normalizing business data ########################
    
    print('######################## Normalizing business data ########################')
    
    def get_attribute_count(x):
        count = 0
        length_x = len(x)
        for i in range(1, len(x)):
            if x[i] == 1:
                count += 1
        return (x[0], sqrt(count))
    
    
    attributes_count_per_business = restaurants_binary_categories_mtl_dataframe.rdd.map(lambda x : get_attribute_count(x)).toDF()
    attributes_count_per_business = attributes_count_per_business.withColumnRenamed('_1','bid').withColumnRenamed('_2','normalized_attribute_count')

    normalized_restaurants_dataframe = restaurants_binary_categories_mtl_dataframe.join(attributes_count_per_business, 'bid')

    def get_normalized_row(x):
        row_dict = x.asDict()
        normalized_attribute_count = row_dict.pop('normalized_attribute_count')
        row_dict.update((k, v/normalized_attribute_count) for k,v in row_dict.items() if k != 'bid')
        output = Row(**row_dict)
        return output
        
    
    normalized_restaurants_dataframe = normalized_restaurants_dataframe.rdd.map(lambda x: get_normalized_row(x)).toDF()
    normalized_restaurants_dataframe.persist(pyspark.StorageLevel.DISK_ONLY)
    
    ############################# Generatng user profiles #########################
    
    print('############################# Generatng user profiles #########################')
    
    (training_reviews_data, test_reviews_data) = reviews_dataframe.randomSplit([0.8,0.2], seed = 42)
         
    reviews_business_categories_dataframe = training_reviews_data.join(normalized_restaurants_dataframe, 'bid')
    
    def user_profile_vector_product(x):
        row_dict = x.asDict()
        stars = row_dict['stars']
        
        # map rating to 1 if > 2, -1 otherwise
        scaled_stars = stars
        scaled_stars = -1
        
        if stars > 2:
            scaled_stars = 1
        
        row_dict.update((k,v*scaled_stars) for k,v in row_dict.items() if k != 'bid' and k != 'stars' and k != 'uid')
        output = Row(**row_dict)
        return output
    
    reviews_business_categories_dataframe = reviews_business_categories_dataframe.rdd.map(lambda x: user_profile_vector_product(x)).toDF()
    user_profile_df = reviews_business_categories_dataframe.groupBy('uid').sum()
    user_profile_df = user_profile_df.drop('sum(uid)', 'sum(stars)', 'sum(bid)')
    user_profile_df = user_profile_df.rdd.filter(lambda x : sum(x[1:-1]) != 0).toDF()
    user_profile_df.persist(pyspark.StorageLevel.DISK_ONLY)
    
    ################ weighted_score = dot product of restaurants and idf score ######################
    
    print('################ weighted_score ######################')
          
    # weighted_score = dot product of restaurants and idf score
    def get_weighted_score(x, category_idf_score):
        row_dict = x.asDict()
        for k,v in row_dict.items():
            if(k != 'bid'):
                row_dict.update({k : v * category_idf_score[k]})
        output = Row(**row_dict)
        return output
    
    weighted_score_df = normalized_restaurants_dataframe.rdd.map(lambda x: get_weighted_score(x, category_idf_score)).toDF()
    
    
    ################################## User predictions #######################################
    
    print('#################### Model predictions #####################')
    
    #predictions = reviews_dataframe.join(user_profile_df, 'uid')
    predictions = test_reviews_data.join(user_profile_df, 'uid')
    predictions = predictions.join(weighted_score_df, 'bid')
    predictions.persist(pyspark.StorageLevel.DISK_ONLY)
    
    def get_dot_product(x, categories_set):
        row_dict = x.asDict()
        bid = row_dict.pop('bid')
        uid = row_dict.pop('uid')
        stars = row_dict.pop('stars')
        
        weighted_score_squared_sum = 0
        user_profile_squared_sum = 0
        
        prediction = 0
        
        for category in categories_set:
            a = row_dict[category]
            weighted_score_squared_sum = weighted_score_squared_sum + (a**2)
            b = row_dict['sum(' + category + ')']
            user_profile_squared_sum = user_profile_squared_sum + (b**2)
            prediction = prediction + (a * b)
            
        #scaled prediction between 1 to 5
        weighted_score_squared_sum = sqrt(weighted_score_squared_sum)
        user_profile_squared_sum = sqrt(user_profile_squared_sum)
        prediction = prediction / (weighted_score_squared_sum * user_profile_squared_sum)
        
        # scale predictions to [1,5]
        prediction = (((prediction - (-1))/(1 - (-1))) * (5 - 1)) + 1
        output = {}
        output['uid'] = uid
        output['bid'] = bid
        output['stars'] = stars
        output['prediction'] = prediction
        output = Row(**output)
        return output
        
    predictions = predictions.rdd.map(lambda x : get_dot_product(x, categories_set)).toDF()
    predictions.cache()
    
    rmse_evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='stars', metricName='rmse')
    mae_evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='stars', metricName='mae')

    
    rmse = rmse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)

    print("RMSE: {}".format(rmse))
    print("MAE: {}".format(mae))
    
    print('#################### predictions for user id 0 #####################')
    
    # user id 0's profile
    uid_0_user_profile = user_profile_df.where('uid == 0')
    
    #  Get all businesses corresponding to the required user and make list of those businesses
    bid_with_our_user = reviews_dataframe.where("uid == 0")
    list_bid_our_user = bid_with_our_user.select('bid').collect()
    list_bid_our_user_modified = []
    for item in list_bid_our_user:
        list_bid_our_user_modified.append(item.bid)
        
    rest_business_weighted_score_df = weighted_score_df.rdd.filter(lambda x: x.bid not in list_bid_our_user_modified).toDF()

    user_0_predictions = rest_business_weighted_score_df.crossJoin(uid_0_user_profile)
    
    def get_dot_product_user_0(x, categories_set):
        row_dict = x.asDict()
        bid = row_dict.pop('bid')
        uid = row_dict.pop('uid')
        
        weighted_score_squared_sum = 0
        user_profile_squared_sum = 0
        
        prediction = 0
        
        for category in categories_set:
            a = row_dict[category]
            weighted_score_squared_sum = weighted_score_squared_sum + (a**2)
            b = row_dict['sum(' + category + ')']
            user_profile_squared_sum = user_profile_squared_sum + (b**2)
            prediction = prediction + (a * b)
            
        #scaled prediction between 1 to 5
        weighted_score_squared_sum = sqrt(weighted_score_squared_sum)
        user_profile_squared_sum = sqrt(user_profile_squared_sum)
        prediction = prediction / (weighted_score_squared_sum * user_profile_squared_sum)
        prediction = (((prediction - (-1))/(1 - (-1))) * (5 - 1)) + 1
        output = {}
        output['uid'] = uid
        output['bid'] = bid
        output['prediction'] = prediction
        output = Row(**output)
        return output
        
    user_0_predictions = user_0_predictions.rdd.map(lambda x : get_dot_product_user_0(x, categories_set)).toDF()
    user_0_predictions = user_0_predictions.join(business_mtl_dataframe, 'bid').orderBy('prediction', ascending=False)
    user_0_predictions = user_0_predictions.select('uid', 'bid', 'name', 'prediction','review_count', 'stars','categories')
    user_0_predictions = user_0_predictions.withColumnRenamed('stars', 'business_rating')
    user_0_predictions.cache()
    user_0_predictions.show(10)
    
    content_results = spark.createDataFrame(user_0_predictions.take(20)).toPandas()
    
    
    ####################### category distribution for user 0's restaurant recommendations ###########
    a = content_results['categories'].str.split(",", n = 0)

    categories = []
    for x in a.values:
        categories.extend([category.lstrip() for category in x])
    

    import pandas as pd

    categories_count = pd.value_counts(categories, sort = True)
    
    categories_count = categories_count.drop("Restaurants") 

    category_labels = list(categories_count.index)
    categories_count.plot(kind = 'bar', rot = 0, figsize=(15, 10), fontsize = 10)
    plt.xticks(range(len(category_labels)), category_labels)
    plt.title('Predicted Restaurants Category Distribution')
    plt.xlabel('Categories')
    plt.ylabel('Categories Count')
    
    
    # rename columns in user profile
    def rename_columns(x):
        row_dict = x.asDict()
        result_dict = {}
        result_dict['uid'] = row_dict['uid']
        for key,value in row_dict.items():
            if key != 'uid':
                key = key[4:-1]
                result_dict[key] = value
        
        output = Row(**result_dict)
        return output
    
    uid_0_user_profile_pandas = uid_0_user_profile.rdd.map(lambda x: rename_columns(x)).toDF().toPandas() 