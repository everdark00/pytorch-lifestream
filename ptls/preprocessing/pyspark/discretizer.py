from typing import List

import pyspark
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

from ptls.preprocessing.base.transformation.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.pyspark.col_transformer import ColTransformerPysparkMixin

class ColNumericDiscretizer(ColTransformerPysparkMixin, ColCategoryTransformer):
    def __init__(self, 
                 col_name_original: str,
                 col_name_target: str = None,
                 is_drop_original_col: bool = True,
                 discr_type: str = 'quantile',
                 n_bins: int = 100,
                 categorical_transformation: List[ColCategoryTransformer] = None
        ):
        '''
        d_type should be one of {'quantile', 'uniform', ‘kmeans’}
        '''
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=col_name_target,
            is_drop_original_col=is_drop_original_col,
        )

        self.discr_type = discr_type
        self.n_bins = n_bins
        self.cat_transformer = categorical_transformation

        if self.discr_type == 'quantile':
            self.discretizer = QuantileDiscretizer(
                numBuckets=n_bins,
                inputCol=self.col_name_target,
                outputCol=self.col_name_target,
                handleInvalid="skip"
            )
        elif self.discr_type == 'kmeans':
            self.discretizer = KMeans(
                k=self.n_bins, 
                featuresCol=self.col_name_target, 
                predictionCol=self.col_name_target,
                seed=42,                 
            )
        else:
            raise AttributeError(
                    f"incorrect discretization type: "
                    f'`discr_type` = "{self.discr_type}" '
                )

    def get_col(self, x: pyspark.sql.DataFrame):
        return x.withColumn(
            self.col_name_target,
            F.col(self.col_name_original),
        )

    def fit(self, x: pyspark.sql.DataFrame):
        super().fit(x)

        df = self.get_col(x)

        if self.discr_type == 'kmeans':
            va = VectorAssembler(inputCols=[self.col_name_target], outputCol=self.col_name_target)
            df = va.transform(df)

        self.discretizer.fit(df)
        self.cat_transformer.fit(self.discretizer.transform(x.select(self.col_name_target)))

        return self

    def transform(self, x: pyspark.sql.DataFrame):
        df = self.get_col(x)

        df = self.cat_transformer.transform(self.discretizer.transform(df))
        df = super().transform(df)
        return df