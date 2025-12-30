from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("spark-ok")
    .getOrCreate()
)

spark = SparkSession.builder.master("local[*]").appName("spark-ok").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print(spark.range(5).collect())

spark.stop()