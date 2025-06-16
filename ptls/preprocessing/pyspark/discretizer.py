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
                inputCol=self.col_name_original,
                outputCol=self.col_name_target,
                handleInvalid="skip"
            )
        elif self.discr_type == 'kmeans':
            self.discretizer = KMeans(
                k=self.n_bins, 
                featuresCol=f'{self.col_name_original}_wrapped', 
                predictionCol=self.col_name_target,
                seed=42,                 
            )
        else:
            raise AttributeError(
                    f"incorrect discretization type: "
                    f'`discr_type` = "{self.discr_type}" '
                )

        self.fitted_discretizer = None

    def get_col(self, x: pyspark.sql.DataFrame):
        if self.discr_type == 'kmeans':
            va = VectorAssembler(inputCols=[self.col_name_original], outputCol=f'{self.col_name_original}_wrapped')
            x = va.transform(x)
        return x

    def fit(self, x: pyspark.sql.DataFrame):
        super().fit(x)
        df = self.get_col(x)

        self.fitted_discretizer = self.discretizer.fit(df)
        self.cat_transformer.fit(self.fitted_discretizer.transform(df))
        return self

    def transform(self, x: pyspark.sql.DataFrame):
        df = self.get_col(x)

        if self.fitted_discretizer is None:
             raise RuntimeError(
                    f"Discretizer instance must be fit before applying transformation!"
                )           
        df = self.cat_transformer.transform(self.fitted_discretizer.transform(df))
        if self.discr_type == 'kmeans':
            df = df.drop(f'{self.col_name_original}_wrapped')
        df = super().transform(df)
        return df