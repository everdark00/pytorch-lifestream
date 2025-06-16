from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import QuantileDiscretizer
from ptls.preprocessing import PysparkDataPreprocessor
from scipy.stats import chisquare
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

def test_add_replace_col():
    np.random.seed(42)
    num_rows = 1000
    spark = SparkSession.builder.getOrCreate()
    df = pd.DataFrame({
        'id': np.random.randint(1, 4, size=num_rows),
        'event_dt': np.random.randint(1, 100, size=num_rows),
        'num_value': np.random.normal(loc=0, scale=100, size=num_rows)
    })
    df_spark = spark.createDataFrame(df)

    n_bins_discr = 10
    discr_type = 'quantile'
    preprocessor =  PysparkDataPreprocessor(
            col_id='id',
            col_event_time='event_dt',
            event_time_transformation='none',
            category_transformation = 'none',
            cols_discretize={'num_value' : (discr_type,  n_bins_discr)},
            cols_numerical=['num_value'],
        )
    processed = preprocessor.fit_transform(df_spark)

    assert 'num_value' in processed.columns, f"Original numeric expected in preprocessed data but not found"
    assert 'num_value_cat' in processed.columns, f"Discretized column 'num_value_cat' expected in preprocessed data but not found"

    preprocessor =  PysparkDataPreprocessor(
            col_id='id',
            col_event_time='event_dt',
            event_time_transformation='none',
            category_transformation = 'none',
            cols_discretize={'num_value' : (discr_type,  n_bins_discr)},
        )
    processed = preprocessor.fit_transform(df_spark)

    assert 'num_value' not in processed.columns, f"Original numeric not expected in preprocessed data but found"
    assert 'num_value_cat' in processed.columns, f"Discretized column expected in preprocessed data but not found"

def test_distribution():
    spark = SparkSession.builder.getOrCreate()
    for discr_type in ['quantile', 'kmeans']:
        np.random.seed(42)
        num_rows = 10000
        df = pd.DataFrame({
            'id': np.random.randint(1, 4, size=num_rows),
            'event_dt': np.random.randint(1, 100, size=num_rows),
            'num_value': np.random.normal(loc=0, scale=100, size=num_rows)
        })
        df_spark = spark.createDataFrame(df)

        n_bins_discr = 100
        preprocessor =  PysparkDataPreprocessor(
                col_id='id',
                col_event_time='event_dt',
                event_time_transformation='none',
                category_transformation = 'frequency',
                cols_discretize={'num_value' : (discr_type,  n_bins_discr)},
            )
        processed = preprocessor.fit_transform(df_spark)
        preproc_cats = processed.select('num_value_cat').toPandas()
        preproc_cats = np.concatenate(list(map(np.array, preproc_cats.num_value_cat)))

        if discr_type == 'quantile':
            kbins = QuantileDiscretizer(
                numBuckets=n_bins_discr,
                inputCol='num_value',
                outputCol='num_value_cat',
                handleInvalid="skip"
            )
            wrapped = df_spark
        else:
            kbins = KMeans(
                    k=n_bins_discr, 
                    featuresCol='num_value_wrapped', 
                    predictionCol='num_value_cat',
                    seed=42,                 
                )
            va = VectorAssembler(inputCols=['num_value'], outputCol='num_value_wrapped')
            wrapped = va.transform(df_spark)
        processed = kbins.fit(wrapped).transform(wrapped)
        kbins_cats = processed.select('num_value_cat').toPandas()
        kbins_cats = kbins_cats.num_value_cat.values.astype(int)

        preproc_counts = np.bincount(preproc_cats - 1, minlength=n_bins_discr)
        kbins_counts = sorted(np.bincount(kbins_cats, minlength=n_bins_discr))[::-1]

        stat, p_value = chisquare(f_obs=preproc_counts, f_exp=kbins_counts)
        print(f"Discretization {discr_type}: chi-square p-value is {p_value:.5f}")

        assert p_value > 0.05, f"Discretization {discr_type}: distributions differ significantly (p={p_value:.5f})"
