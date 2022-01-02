##############################################################################
# PySpark Machine Learning: Binary Classification                            #
# by Davian Ricardo Chin                                                     #
##############################################################################

# SETTING UP THE ENVIRONMENT FOR PYSPARK PROGRAMMING
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySpark Machine Learning: Binary Classification").getOrCreate()

# UNSW-NB15 DATASET: CLEANED VERSION
unsw_data = spark.read.csv("hdfs://localhost:9000/tmp/exported/clean_data/network_data_final", 
                           inferSchema=True).toDF("srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
                                                  "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
                                                  "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt",
                                                  "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
                                                  "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
                                                  "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label")

# UNSW-NB15 DATASET WITH ONLY SELECTED FEATURES
unsw_net_data = unsw_data.select("state", "service", "sttl", "swin", "dwin", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
                                 "ct_dst_sport_ltm", "ct_dst_src_ltm", "label", "attack_cat")

# FEATURE SELECTION AND VECTORISARION
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml.feature import PCA

categorical_cols = ["state", "service", "attack_cat"]
indexed_cols = ["state_index", "service_index", "attack_cat_index"]

indexed_unsw_net_data = StringIndexer(inputCols=categorical_cols, outputCols=indexed_cols).fit(unsw_net_data).transform(unsw_net_data)

vectorised_unsw_net_data = VectorAssembler(inputCols=["state_index", "service_index", "sttl", "swin", "dwin", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
                                                      "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat_index"],
                                           outputCol="features").transform(indexed_unsw_net_data)

unsw_net_data_final = vectorised_unsw_net_data.select("features", "label", "attack_cat_index")

unsw_net_data_scaled = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                                      withStd=True, withMean=True).fit(unsw_net_data_final).transform(unsw_net_data_final)

# PCA DIMENSIONALITY REDUCTION
col_names = unsw_net_data_scaled.columns
features_rdd = unsw_net_data_scaled.rdd.map(lambda x:x[0:]).toDF(col_names)


pca = PCA(k=5, inputCol="scaledFeatures", outputCol="pcaFeatures")
pca_reduced_unsw_data = pca.fit(features_rdd).transform(features_rdd).select('pcaFeatures', 'label', 'attack_cat_index')

unsw_bc_data = pca_reduced_unsw_data.select('pcaFeatures', 'label').toDF('features','label')


# SETTING THE STAGE FOR MACHINE LEARNING
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# SPLITIING THE DATA INTO TRAINING AND TESTING DATSETS
train, test = unsw_bc_data.randomSplit([0.7, 0.3], seed=25)

lr = LogisticRegression(maxIter=10)
lrm = lr.fit(train)

# VIEWING SOME STATISTICAL DATA ABOUT THE LOGISTICS REGRESSION CLASSIFIER
lrm.summary.predictions.show()
lrm.summary.predictions.describe().show()

# APPLYING THE TRAINED LOGISTICS REGRESSION CLASSIFIER TO THE TEST DATA
lr_pred = lrm.evaluate(test)
#lr_pred.predictions.show(5)

# EVALUATING THE LOGISTICS REGRESSION MODEL FOR ACCURACY USING AUC MEASURE
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
auc = evaluator.evaluate(lr_pred.predictions)
print()
print("LR Classifier Accuracy Score: {}".format(auc))
# Prints out: LR Classifier Accuracy Score: 0.99648340528807

# APPLYING AND EVALUATING A CROSS VALIDATION MODEL WITH LOGISTICS REGRESSION CLASSIFIER
# Create a parameter grid for the cross validator
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).addGrid(lr.elasticNetParam, [0.2, 0.6, 0.8]).build()

# Train the cross validator estimator to the training data
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=3)

cvm = cv.fit(train)
cvpred = cvm.transform(test)

# Evaluate the cross validation model using the BinaryClassificationEvaluator
cv_result = evaluator.evaluate(cvpred)
print()
print("Cross Validation Accuracy Score for LR Classifier: {}".format(cv_result))
# Prints out: Cross Validation Accuracy Score for LR Classifier: 0.9889961234642317
