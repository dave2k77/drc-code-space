##############################################################################
# PySpark Advanced Analytics: Analytics for Feature Selection                #
# by Davian Ricardo Chin                                                     #
##############################################################################

#=====================================================================================
# SETTING UP THE ENVIRONMENT FOR PYSPARK PROGRAMMING
#=====================================================================================

# import SparkSession
#=====================================================
from pyspark.sql import SparkSession

# Set up a spark session
spark = SparkSession.builder.getOrCreate()

# LOADING DATASETS INTO SPARK DATAFRAMES: DATASETS LOCATION
#============================================================
categorical_url = '/home/davianc/Documents/cyberattack_data/data_tables/categorical_data'
discrete_url = '/home/davianc/Documents/cyberattack_data/data_tables/discrete_data'
continuous_url = '/home/davianc/Documents/cyberattack_data/data_tables/continuous_data'
unsw_data_url = '/home/davianc/Documents/cyberattack_data/data_tables/network_data_final'

#=====================================================================================
# CATEGORICAL FEATURES OF THE UNSW-NB15 DATASET
#=====================================================================================

categorical_data = spark.read.csv(categorical_url, inferSchema=True).toDF("srcip", "dstip", "proto", "state",
                                                                          "service", "stime", "ltime", "attack_cat")

from pyspark.ml.feature import StringIndexer, VectorAssembler

# Index categorical features using StandardScaler
#=================================================
categorical_indexer = StringIndexer(inputCols=['srcip', 'dstip', 'proto', 'state', 'service', 'attack_cat'],
                                    outputCols=['srcip_index', 'dstip_index', 'proto_index', 'state_index',
                                                'service_index', 'attack_cat_index'])
indexed_categorical_data = categorical_indexer.fit(categorical_data).transform(categorical_data)

# Create categorical features vector using VectorAssembler
#==========================================================
categorical_assember = VectorAssembler(inputCols=['srcip_index', 'dstip_index', 'proto_index', 'state_index',
                                                  'service_index', 'attack_cat_index'],
                                       outputCol='categorical_features')
categorical_data_transformed = categorical_assember.transform(indexed_categorical_data).select('categorical_features',
                                                                                               'attack_cat_index')

# Pearson's ChiSquare Test for Independence
#===========================================
from pyspark.ml.stat import ChiSquareTest

# Applying the ChiSquareTest to the transformed categorical dataset
independenceResult = ChiSquareTest.test(categorical_data_transformed, "categorical_features", "attack_cat_index")

# Storing the test results into variables
degreesOfFreedom = independenceResult.select("degreesOfFreedom").collect()[0]
p_values = independenceResult.select("pValues").collect()[0]
testStatistics = independenceResult.select("statistics").collect()[0]

# Printing out the test results
#===============================================================================
print("Pearson's ChiSquare Test of Independence Results")
print("=======================================================================")
print("Degrees of Freedom: {}".format(degreesOfFreedom))
print("=======================================================================")
print("P-Values: {}".format(p_values))
print("=======================================================================")
print("Test Statistics: {}".format(testStatistics))
print("=======================================================================")

categorical_features_final = categorical_data.select("srcip", "dstip", "proto", "state", "service", "attack_cat")

# Checking the  number of distinct categories in categorical data (drop those with number greater than 32)
srcip_distinct = categorical_features_final.select("srcip").distinct().count() # num = 43 > 32 : drop
dstip_distinct = categorical_features_final.select("dstip").distinct().count() # num = 47 > 32 : drop
proto_distinct = categorical_features_final.select("proto").distinct().count() # num = 134 > 32 : drop
state_distinct = categorical_features_final.select("state").distinct().count() # num = 16 < 32 : keep
service_distinct = categorical_features_final.select("service").distinct().count() # num = 2 < 32 : keep
categorical_features_final.select("attack_cat").distinct().count() # label : keep 

# Results of Distinct Counts
#===================================================================================
print("Distinct Category Counts for Categorical Features (num > 32: drop)")
print("======================================================================")
print("Number of distinct groupings for srcip: {}".format(srcip_distinct))
print("======================================================================")
print("Number of distinct groupings for dstip: {}".format(dstip_distinct))
print("======================================================================")
print("Number of distinct groupings for proto: {}".format(proto_distinct))
print("======================================================================")
print("Number of distinct groupings for state: {}".format(state_distinct))
print("======================================================================")
print("Number of distinct groupings for service: {}".format(service_distinct))
print("======================================================================")

#=====================================================================================
# DISCRETE FEATURES OF THE UNSW-NB15 DATASET
#=====================================================================================

discrete_data = spark.read.csv(discrete_url, 
                               inferSchema=True).toDF("sport","dsport","sbytes","dbytes","sttl","dttl",
                                                      "sloss","dloss","spkts","dpkts","swin","dwin","stcpb",
                                                      "dtcpb","smeansz","dmeansz","trans_depth","res_bdy_len",
                                                      "is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd",
                                                      "is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst",
                                                      "ct_dst_ltm","ct_src_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
                                                      "ct_dst_src_ltm","label")

# Assemble features into a column of features vectors
discrete_assember = VectorAssembler(inputCols=["sport","dsport","sbytes","dbytes","sttl","dttl","sloss","dloss","spkts","dpkts","swin","dwin",
                                               "stcpb","dtcpb","smeansz","dmeansz","trans_depth","res_bdy_len","is_sm_ips_ports","ct_state_ttl",
                                               "ct_flw_http_mthd","is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm",
                                               "ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm","label"],
                                    outputCol="discreteFeatures")

discrete_data_vectors = discrete_assember.transform(discrete_data).select("discreteFeatures", 'label')


# Correlation Analysis for Discrete Features
#=============================================
from pyspark.ml.stat import Correlation

# Computing the correlation matrix
#===================================
discrete_matrix = Correlation.corr(discrete_data_vectors,'discreteFeatures').collect()[0][0].toArray().tolist()

discrete_matrix_df = spark.createDataFrame(discrete_matrix, ["sport","dsport","sbytes","dbytes","sttl","dttl","sloss","dloss",
                                                             "spkts","dpkts","swin","dwin","stcpb","dtcpb","smeansz","dmeansz",
                                                             "trans_depth","res_bdy_len","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd",
                                                             "is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm",
                                                             "ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm","label"])

# Loading the discrete correlation dataset
#=============================================
discrete_col_corr = spark.read.csv('/home/davianc/Documents/cyberattack_data/data_tables/col_corr.csv',
                                   inferSchema=True, header=True)

# Dropping irrelevant feature columns using PySpark SQL Query
#===============================================================
discrete_col_corr.createOrReplaceTempView("DATA")
result = spark.sql("SELECT name, corr FROM DATA WHERE corr >= 0.25 OR corr <= -0.25")
discrete_cols_final = result.select("name")


#=====================================================================================
# CONTINUOUS FEATURES OF THE UNSW-NB15 DATASET
#=====================================================================================

# Analyse the continuous features of the dataset
continuous_data = spark.read.csv(continuous_url,
                                 inferSchema=True).toDF("dur", "sload", "dload", "sjit", "djit", "sintpkt","dintpkt",
                                                                        "tcprtt", "synack", "ackdat")

# DESCRIPTIVE ANALYTICS AND KERNEL DENSITY ESTIMATION
#===================================================================

# Descriptive Analytics
#=======================
continuous_data.describe().show(5)


# Kernel Density Plot: tcprtt feature column
#=============================================
tcprtt_data = continuous_data.select("tcprtt")
tcprtt_data_pd = tcprtt_data.toPandas()
tcprtt_data_pd.plot.kde(bw_method=3)

# Loading the UNSW-NB15 Dataset
unsw_data = spark.read.csv(unsw_data_url,
                           inferSchema=True).toDF("srcip", "sport", "dstip", "dsport", "proto",
                                                  "state", "dur", "sbytes","dbytes", "sttl", "dttl",
                                                  "sloss", "dloss", "service", "sload", "dload",
                                                  "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
                                                  "smeansz", "dmeansz","trans_depth", "res_bdy_len",
                                                  "sjit", "djit", "stime", "ltime", "sintpkt","dintpkt",
                                                  "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
                                                  "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
                                                  "ct_srv_dst","ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
                                                  "ct_dst_sport_ltm","ct_dst_src_ltm", "attack_cat", "label")

# Computing Correlations with the 'label' Column
#=================================================
dur_label_cor = unsw_data.select("dur", "label").corr('dur', 'label', method='pearson') 
# poor correlation with 'label' and relatively low variance (high bias) : drop

sjit_label_cor = unsw_data.select("sjit", "label").corr('sjit', 'label', method='pearson') 
# poor correlation with 'label' : drop

djit_label_cor = unsw_data.select("djit", "label").corr('djit', 'label', method='pearson') 
# poor correlation with 'label': drop

sload_label_cor = unsw_data.select("sload", "label").corr('sload', 'label', method='pearson') 
# poor correlation with 'label': drop

dload_label_cor = unsw_data.select("dload", "label").corr('dload', 'label', method='pearson') 
# poor correlation with 'label' : drop

tcprtt_label_cor = unsw_data.select("tcprtt", "label").corr('tcprtt', 'label', method='pearson')

tcprtt_synack_cor = unsw_data.select("tcprtt", "synack").corr('tcprtt', 'synack', method='pearson')

tcprtt_ackdat_cor = unsw_data.select("tcprtt", "ackdat").corr('tcprtt', 'ackdat', method='pearson')

# Print Correlation Results
#============================================================================================================
print("Correlation Results : corr in  (-0.25, 0.25): drop")
print("====================================================================================================")
print("CORR(dur, label): {}".format(dur_label_cor)) # drop
print("====================================================================================================") 
print("CORR(sjit, label): {}".format(sjit_label_cor))  # drop
print("====================================================================================================")
print("CORR(djit, label): {}".format(djit_label_cor)) # drop
print("====================================================================================================")
print("CORR(sload, label): {}".format(sload_label_cor)) # drop
print("====================================================================================================")
print("CORR(dload, label): {}".format(dload_label_cor)) # drop
print("====================================================================================================")
print("CORR(tcprtt, label): {}".format(tcprtt_label_cor)) # high correlation with synack and ackdat : keep
print("====================================================================================================")
print("CORR(tcprtt, synack): {}".format(tcprtt_synack_cor)) # drop
print("====================================================================================================")
print("CORR(tcprtt, ackdat): {}".format(tcprtt_ackdat_cor)) # drop
print("====================================================================================================")

# FEATURE SELECTION AND FINAL DATASET
#========================================
unsw_net_data = unsw_data.select("state", "service", "sttl", "swin", "dwin", "ct_state_ttl", "ct_srv_src", "ct_srv_dst",
                                 "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
                                 "tcprtt", "label", "attack_cat")
