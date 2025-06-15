from pyspark.sql import SparkSession
import sys
print("PYTHON:", sys.executable)

if __name__ == '__main__':
    spark = SparkSession.builder.appName("QuantileBinning").getOrCreate()

    df = spark.createDataFrame([(1, "Alice"), (2, "Bob")], ["id", "name"])
    print(df.collect())

    spark.stop()