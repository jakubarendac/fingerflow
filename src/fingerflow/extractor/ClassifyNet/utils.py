import pandas as pd


def format_classified_data(numpy_data):
    if numpy_data.size == 0:
        return pd.DataFrame([], columns=['x', 'y', 'angle', 'score', 'class'])

    return pd.DataFrame(numpy_data, columns=['x', 'y', 'angle', 'score', 'class'])
