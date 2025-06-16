from ptls.preprocessing import PandasDataPreprocessor
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chisquare
import numpy as np
import pandas as pd
import torch

def test_add_replace_col():
    np.random.seed(42)
    num_rows = 1000
    df = pd.DataFrame({
        'id': np.random.randint(1, 4, size=num_rows),
        'event_dt': np.random.randint(1, 100, size=num_rows),
        'num_value': np.random.normal(loc=0, scale=100, size=num_rows)
    })

    n_bins_discr = 10
    discr_type = 'quantile'
    preprocessor =  PandasDataPreprocessor(
            col_id='id',
            col_event_time='event_dt',
            event_time_transformation='none',
            category_transformation = 'none',
            cols_discretize={'num_value' : (discr_type,  n_bins_discr)},
            cols_numerical=['num_value'],
            return_records=True,
        )
    processed = preprocessor.fit_transform(df)

    assert 'num_value' in processed[0], f"Original numeric expected in preprocessed data but not found"
    assert 'num_value_cat' in processed[0], f"Discretized column 'num_value_cat' expected in preprocessed data but not found"

    preprocessor =  PandasDataPreprocessor(
            col_id='id',
            col_event_time='event_dt',
            event_time_transformation='none',
            category_transformation = 'none',
            cols_discretize={'num_value' : (discr_type,  n_bins_discr)},
            return_records=True,
        )
    processed = preprocessor.fit_transform(df)

    assert 'num_value' not in processed[0], f"Original numeric not expected in preprocessed data but found"
    assert 'num_value_cat' in processed[0], f"Discretized column expected in preprocessed data but not found"

def test_distribution():
    for discr_type in ['kmeans', 'quantile']:
        np.random.seed(42)
        num_rows = 10000
        df = pd.DataFrame({
            'id': np.random.randint(1, 4, size=num_rows),
            'event_dt': np.random.randint(1, 100, size=num_rows),
            'num_value': np.random.normal(loc=0, scale=100, size=num_rows)
        })

        n_bins_discr = 100
        preprocessor =  PandasDataPreprocessor(
                col_id='id',
                col_event_time='event_dt',
                event_time_transformation='none',
                category_transformation = 'frequency',
                cols_discretize={'num_value' : (discr_type,  n_bins_discr)},
                return_records=True,
            )
        processed = preprocessor.fit_transform(df)
        preproc_cats = torch.cat([x['num_value_cat'] for x in processed]).numpy()

        kbins = KBinsDiscretizer(n_bins=n_bins_discr, encode='ordinal', strategy=discr_type)
        sklearn_cats = kbins.fit_transform(df[['num_value']]).astype(int).flatten()

        preproc_counts = np.bincount(preproc_cats - 1, minlength=n_bins_discr)
        sklearn_counts = sorted(np.bincount(sklearn_cats, minlength=n_bins_discr))[::-1]

        stat, p_value = chisquare(f_obs=preproc_counts, f_exp=sklearn_counts)
        print(f"Discretization {discr_type}: chi-square p-value is {p_value:.5f}")

        assert p_value > 0.05, f"Discretization {discr_type}: distributions differ significantly (p={p_value:.5f})"