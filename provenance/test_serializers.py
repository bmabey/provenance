import pandas as pd

import provenance.serializers as s


def test_default_object_serializers():
    assert s.object_serializer('foo') == 'joblib'
    assert s.object_serializer((1, 2, 3)) == 'joblib'
    assert s.object_serializer({'foo': 42}) == 'joblib'

    df = pd.DataFrame([{'foo': 42}, {'foo': 55}])
    assert s.object_serializer(df) == 'pd_df_parquet'

    series = df.foo
    assert s.object_serializer(series) == 'pd_series_parquet'
