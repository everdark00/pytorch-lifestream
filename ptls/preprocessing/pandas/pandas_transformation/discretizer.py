from typing import List
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer

from ptls.preprocessing.base.transformation.col_category_transformer import ColCategoryTransformer

class ColNumericDiscretizer(ColCategoryTransformer):
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

        if self.discr_type not in {'quantile', 'kmeans'}:
            raise AttributeError(
                    f"incorrect discretization type: "
                    f'`discr_type` = "{self.discr_type}" '
                )

        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, 
            encode='ordinal', 
            strategy=self.discr_type,
 #           quantile_method='averaged_inverted_cdf'
        )
    
    def __repr__(self):
        return "Num2Cat transformation"

    def fit(self, x: pd.DataFrame):
        super().fit(x)
        x = self.attach_column(x, x[self.col_name_original].rename(self.col_name_target))

        self.discretizer.fit(x.loc[:, [self.col_name_target]])
        self.cat_transformer.fit(
            pd.DataFrame(self.discretizer.transform(x.loc[:, [self.col_name_target]]).astype('int'), 
                         columns=[self.col_name_target]
            )
        )
        return self

    def transform(self, x: pd.DataFrame):
        x = self.attach_column(x, x[self.col_name_original].rename(self.col_name_target))
        x[self.col_name_target] = self.cat_transformer.transform(
            pd.DataFrame(self.discretizer.transform(x[[self.col_name_target]]).astype('int'),
                         columns=[self.col_name_target]
            )
        )
        x = super().transform(x)
        return x