import os
import subprocess
from pdb import set_trace


os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-11.0.22"
os.environ["SPARK_HOME"] = r"C:\Users\toppc\spark"
os.environ["HADOOP_HOME"] = r"C:\Users\toppc\hadoop-3.0.0"
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PATH"] = os.pathsep.join([
    os.path.join(os.environ["SPARK_HOME"], "bin"),
    os.path.join(os.environ["HADOOP_HOME"], "bin"),
    os.environ["PATH"],
])
spark_submit_path = os.path.join(os.environ["SPARK_HOME"], "bin", "spark-submit.cmd")

cmd = r'"C:\Users\toppc\spark\bin\spark-submit.cmd --version'
#subprocess.run(["echo", "%PATH%"], env=os.environ, shell=True)
#result = subprocess.run(cmd, env=os.environ, shell=True, capture_output=True, text=True)

cmd = [
    r"C:\Users\toppc\spark\bin\spark-submit.cmd",
    "--master", "local[8]",
    "--name", "Gender Make Dataset",
    "--driver-memory", "16G",
    "--conf", "spark.sql.shuffle.partitions=60",
    "--conf", "spark.sql.parquet.compression.codec=snappy",
    "--conf", "spark.ui.port=4041",
    "--conf", "spark.local.dir=../../data/.spark_local_dir",
    "make_dataset.py",
    "--data_path", "../../data/gender",
    "--trx_files", "transactions.csv",
    "--col_client_id", "customer_id",
    "--cols_event_time", "#gender", "tr_datetime",
    "--cols_category", "mcc_code", "tr_type", "term_id",
    "--cols_discretize", "amount:quntile#100",
    "--target_files", "gender_train.csv",
    "--col_target", "gender",
    "--test_size", "0.1",
    "--output_train_path", "../../data/gender/train_trx_discr.parquet",
    "--output_test_path", "../../data/gender/test_trx_discr.parquet",
    "--output_test_ids_path", "../../data/gender/test_ids.csv",
    "--print_dataset_info"
]


#subprocess.run(cmd)

set "JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.0.27.6-hotspot"
set "SPARK_HOME=C:\Users\toppc\spark"
set "HADOOP_HOME=C:\Users\toppc\hadoop-3.0.0"
set "PYSPARK_PYTHON=C:\Users\toppc\.conda\envs\ptls-fork\python.exe"
set "SPARK_LOCAL_IP=127.0.0.1"
set "PATH=%SPARK_HOME%\bin;%HADOOP_HOME%\bin;%PATH%"

spark.hadoop.hadoop.security.authorization false
spark.hadoop.fs.permissions.umask-mode 000


spark-submit --master local[4] --name "Gender Make Dataset" --driver-memory 4G --conf spark.sql.shuffle.partitions=60 --conf spark.sql.parquet.compression.codec=snappy --conf spark.ui.port=4041 --conf spark.hadoop.fs.file.impl=org.apache.hadoop.fs.RawLocalFileSystem --conf spark.hadoop.fs.file.impl.disable.cache=true make_datasets_spark.py --data_path ../../data/gender --trx_files transactions.csv --col_client_id customer_id --cols_event_time #gender tr_datetime --cols_category mcc_code tr_type term_id --cols_discretize amount:quantile#100 --target_files gender_train.csv --col_target gender --test_size 0.1 --output_train_path ../../data/gender/train_trx_discr.parquet --output_test_path ../../data/gender/test_trx_discr.parquet --output_test_ids_path ../../data/gender/test_ids.csv --print_dataset_info

spark-submit --master local[8] --name "Gender Make Dataset" --driver-memory 16G --conf spark.sql.shuffle.partitions=60 --conf spark.sql.parquet.compression.codec=snappy --conf spark.ui.port=4041 --conf spark.local.dir=../../data/.spark_local_dir

spark-submit --master local[1]  --verbose --name "Gender Make Dataset"

spark-submit --master local[4] --name "Gender Make Dataset" --driver-memory 4G --conf spark.sql.shuffle.partitions=60 --conf spark.sql.parquet.compression.codec=snappy --conf spark.ui.port=4041 --conf spark.hadoop.hadoop.security.authorization=false --conf spark.hadoop.fs.permissions.umask-mode=000 --conf spark.hadoop.fs.file.impl=org.apache.hadoop.fs.RawLocalFileSystem --conf spark.hadoop.fs.file.impl.disable.cache=true make_datasets_spark.py --data_path ../../data/gender --trx_files transactions.csv --col_client_id customer_id --cols_event_time #gender tr_datetime --cols_category mcc_code tr_type term_id --cols_log_norm amount --target_files gender_train.csv --col_target gender --test_size 0.1 --output_train_path ../../data/gender/train_trx_discr.parquet --output_test_path ../../data/gender/test_trx_discr.parquet --output_test_ids_path ../../data/gender/test_ids.csv --print_dataset_info
