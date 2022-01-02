##############################################################################
# PySpark Machine Learning: Multiclass Classification                        #
# by Davian Ricardo Chin
##############################################################################

# SETTING UP THE ENVIRONMENT FOR PYSPARK PROGRAMMING
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySpark Machine Learning: Binary Classification").getOrCreate()

# UNSW-NB15 DATASET: CLEANED VERSION
unsw_data = spark.read.csv("hdfs://localhost:9000/tmp/exported/clean_data/network_data_final", inferSchema=True).toDF("srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label")

# UNSW-NB15 DATASET WITH ONLY SELECTED FEATURES
unsw_net_data = unsw_data.select("state", "service", "sttl", "swin", "dwin", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "label", "attack_cat")

# FEATURE SELECTION AND VECTORISARION
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml.feature import PCA

categorical_cols = ["state", "service", "attack_cat"]
indexed_cols = ["state_index", "service_index", "attack_cat_index"]

indexed_unsw_net_data = StringIndexer(inputCols=categorical_cols, outputCols=indexed_cols).fit(unsw_net_data).transform(unsw_net_data)

vectorised_unsw_net_data = VectorAssembler(inputCols=["state_index", "service_index", "sttl", "swin", "dwin", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat_index"], outputCol="features").transform(indexed_unsw_net_data)

unsw_net_data_final = vectorised_unsw_net_data.select("features", "label", "attack_cat_index")

unsw_net_data_scaled = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True).fit(unsw_net_data_final).transform(unsw_net_data_final)

# PCA DIMENSIONALITY REDUCTION
col_names = unsw_net_data_scaled.columns
features_rdd = unsw_net_data_scaled.rdd.map(lambda x:x[0:]).toDF(col_names)


pca = PCA(k=5, inputCol="scaledFeatures", outputCol="pcaFeatures")
pca_reduced_unsw_data = pca.fit(features_rdd).transform(features_rdd).select('pcaFeatures', 'label', 'attack_cat_index')

unsw_mc_data = pca_reduced_unsw_data.select('pcaFeatures', 'attack_cat_index').toDF('features','attack_cat_index')

unsw_mc_data.show(n=10, truncate=False)

# SETTING THE STAGE FOR MACHINE LEARNING
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# SPLITIING THE DATA INTO TRAINING AND TESTING DATSETS
train, test = unsw_mc_data.randomSplit([0.7, 0.3], seed=25)
train.show(n=10, truncate=False)
test.show(n=10, truncate=False)

# INSTANTIATING AND TRAINING THE RANDOM FOREST CLASSIFIER
# Instantiate the random forest classifier and train it with the training data
rf = RandomForestClassifier(labelCol="attack_cat_index", featuresCol="features")
rfm = rf.fit(train)

# Analyse the prediction results of the model
# rfm.summary.predictions.show(n=5, truncate=False)

# TESTING THE TRAINED RANDOM FOREST CLASSIFIER ON THE TEST DATA
rf_pred = rfm.transform(test)


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
cvm = cv.fit(train)
cvprediction = cvm.transform(test)

# Evaluate the cross validation model using the MulticlassClassificationEvalutor
cv_acc =acc.evaluate(cvprediction)
print("Cross Validation Accuracy Score: {}".format(cv_acc))
# Prints out: Cross Validation Accuracy Score: 0.9771437414623839
