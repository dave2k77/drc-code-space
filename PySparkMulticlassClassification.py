##############################################################################
# PySpark Machine Learning: Multiclass Classification                        #
# by Davian Ricardo Chin
##############################################################################

# SETTING UP THE ENVIRONMENT FOR PYSPARK PROGRAMMING
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PySpark Machine Learning: Multiclass Classification").getOrCreate()

# LOADING THE UNSW-NB15 NETWORK TRAFFIC DATA
unsw_data_url = '/home/davianc/Documents/cyberattack_data/data_tables/network_data_final'

unsw_data = spark.read.csv(unsw_data_url, inferSchema=True).toDF("srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
                                                                 "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin",
                                                                 "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
                                                                 "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
                                                                 "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
                                                                 "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
                                                                 "attack_cat", "label")

# FEATURE SELECTION AND VECTORISARION
from pyspark.ml.feature import StringIndexer, VectorAssembler

# CREATE A NEW DATAFRAME USING ONLY THE RELEVANT COLUMNS
unsw_net_data = unsw_data.select("state", "service", "sttl", "swin", "dwin", "ct_state_ttl", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                                 "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "tcprtt", "label", "attack_cat")

# INDEX THE CATEGORICAL FEATURE COLUMNS
indexed_unsw_net_data = StringIndexer(inputCols=["state", "service", "sttl", "swin", "dwin", "attack_cat"],
                                      outputCols=["state_index", "service_index", "sttl_index", "swin_index", "dwin_index",
                                                  "attack_cat_index"]).fit(unsw_net_data).transform(unsw_net_data)

# REMOVE UNNECESSARY COLUMNS
indexed_unsw_net_data = indexed_unsw_net_data.drop("state", "service", "sttl", "swin", "dwin", "attack_cat")

# VECTORISE FEATURES COLUMNS
vectorised_unsw_net_data = VectorAssembler(inputCols=['ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
                                                      'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'tcprtt', 'state_index', 'service_index', 'sttl_index',
                                                      'swin_index', 'dwin_index'],
                                           outputCol='features').transform(indexed_unsw_net_data).select("features", "label", "attack_cat_index")

# CREATING FINAL DATAFRAME WITH FEATURES VECTOR AND LABEL COLUMNS
unsw_mc_data = vectorised_unsw_net_data.select("features", "attack_cat_index")

# SETTING THE STAGE FOR MACHINE LEARNING
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StandardScaler

# SPLITIING THE DATA INTO TRAINING AND TESTING DATSETS
train, test = unsw_mc_data.randomSplit([0.7, 0.3], seed=25)

# NORMALISING THE FEATURES WITH THE STANDARDSCALER ESTIMATOR
stc = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaledTrain = stc.fit(train).transform(train)
scaledTest = stc.fit(test).transform(test)

# INSTANTIATING AND TRAINING THE RANDOM FOREST CLASSIFIER
# Instantiate the random forest classifier and train it with the training data
rf = RandomForestClassifier(labelCol="attack_cat_index", featuresCol="features")
rfm = rf.fit(scaledTrain)

# Analyse the prediction results of the model
# rfm.summary.predictions.show(n=5, truncate=False)

# TESTING THE TRAINED RANDOM FOREST CLASSIFIER ON THE TEST DATA
rf_pred = rfm.transform(scaledTest)


# EVALUATING THE ACCURACY OF THE RANDOM FOREST MODEL USING THE MULTICLASSCLASSIFICATIONEVALUATOR
# Instantiate the MulticlassClassificationEvaluator
acc = MulticlassClassificationEvaluator(labelCol="attack_cat_index", predictionCol="prediction", metricName="accuracy")

# Evaluate the accuracy of the random forest classifier
rf_acc = acc.evaluate(rf_pred)
print("RFC Accuracy Score: {}".format(rf_acc))
# Prints out: RFC Accuracy Score: 0.9731022311636752

# APPLYING AND EVALUATING A CROSS VALIDATION MODEL WITH RANDOM FOREST CLASSIFIER
# Create a parameter grid for the cross validator
paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [5, 20, 50]).addGrid(rf.maxDepth, [2, 5, 10]).build()

# Instantiate the cross validator estimator
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=acc, numFolds=3)

# Train the cross validator estimator to the training data
cvm = cv.fit(scaledTrain)
cvprediction = cvm.transform(scaledTest)

# Evaluate the cross validation model using the MulticlassClassificationEvalutor
cv_acc =acc.evaluate(cvprediction)
print("Cross Validation Accuracy Score: {}".format(cv_acc))
# Prints out: Cross Validation Accuracy Score: 0.9771437414623839
