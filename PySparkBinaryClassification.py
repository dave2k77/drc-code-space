##############################################################################
# PySpark Machine Learning: Binary Classification                            #
# by Davian Ricardo Chin                                                     #
##############################################################################

# SETTING UP THE ENVIRONMENT FOR PYSPARK PROGRAMMING
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySpark Machine Learning: Binary Classification").getOrCreate()


# LOADING THE UNSW-NB15 NETWORK TRAFFIC DATA
unsw_data_url = '/home/davianc/Documents/cyberattack_data/data_tables/network_data_final'

unsw_data = spark.read.csv(unsw_data_url, inferSchema=True).toDF("srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
                                                                 "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts",
                                                                 "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
                                                                 "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
                                                                 "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
                                                                 "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
                                                                 "ct_dst_src_ltm", "attack_cat", "label")

# FEATURE SELECTION AND VECTORISARION
from pyspark.ml.feature import StringIndexer, VectorAssembler

# CREATE A NEW DATAFRAME USING ONLY THE RELEVANT COLUMNS
unsw_net_data = unsw_data.select("state", "service", "sttl", "swin", "dwin", "ct_state_ttl", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                                 "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "tcprtt", "label", "attack_cat")

# INDEX THE CATEGORICAL FEATURE COLUMNS
indexed_unsw_net_data = StringIndexer(inputCols=["state", "service", "sttl", "swin", "dwin", "attack_cat"],
                                      outputCols=["state_index", "service_index", "sttl_index", "swin_index", "dwin_index", "attack_cat_index"]).fit(unsw_net_data).transform(unsw_net_data)

# REMOVE UNNECESSARY COLUMNS
indexed_unsw_net_data = indexed_unsw_net_data.drop("state", "service", "sttl", "swin", "dwin", "attack_cat")

# VECTORISE FEATURES COLUMNS
vectorised_unsw_net_data = VectorAssembler(inputCols=['ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
                                                      'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'tcprtt', 'state_index', 'service_index', 'sttl_index',
                                                      'swin_index', 'dwin_index'],
                                           outputCol='features').transform(indexed_unsw_net_data).select("features", "label", "attack_cat_index")

# CREATING FINAL DATAFRAME WITH FEATURES VECTOR AND LABEL COLUMNS
unsw_bc_data = vectorised_unsw_net_data.select("features", "label")

# SETTING THE STAGE FOR MACHINE LEARNING
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StandardScaler

# SPLITIING THE DATA INTO TRAINING AND TESTING DATSETS
train, test = unsw_bc_data.randomSplit([0.7, 0.3], seed=25)

# NORMALISING THE FEATURES WITH THE STANDARDSCALER ESTIMATOR
stc = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)

scaledTrain = stc.fit(train).transform(train)
scaledTest = stc.fit(test).transform(test)

# INSTANTIATING AND TRAINING THE LOGISTICS REGRESSION CLASSIFIER
# Instantiate the logistics regression classifier and train it with the training data
lr = LogisticRegression(maxIter=10)
lrm = lr.fit(scaledTrain)

# VIEWING SOME STATISTICAL DATA ABOUT THE LOGISTICS REGRESSION CLASSIFIER
lrm.summary.predictions.show()
lrm.summary.predictions.describe().show()

# APPLYING THE TRAINED LOGISTICS REGRESSION CLASSIFIER TO THE TEST DATA
lr_pred = lrm.evaluate(scaledTest)
#lr_pred.predictions.show(5)

# EVALUATING THE LOGISTICS REGRESSION MODEL FOR ACCURACY USING AUC MEASURE
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
auc = evaluator.evaluate(lr_pred.predictions)
print("LR Classifier Accuracy Score: {}".format(auc))
# Prints out: LR Classifier Accuracy Score: 0.9362559574812116

# APPLYING AND EVALUATING A CROSS VALIDATION MODEL WITH LOGISTICS REGRESSION CLASSIFIER
# Create a parameter grid for the cross validator
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()

# Train the cross validator estimator to the training data
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=3)

cvm = cv.fit(scaledTrain)
cvpred = cvm.transform(scaledTest)

# Evaluate the cross validation model using the BinaryClassificationEvaluator
cv_result = evaluator.evaluate(cvpred)
print("Cross Validation Accuracy Score for LR Classifier: {}".format(cv_result))
# Prints out: Cross Validation Accuracy Score for LR Classifier: 0.9151539933672815
