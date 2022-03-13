import pandas as pd
import numpy as np


df1 = pd.DataFrame({'user_id': [214, 224, 234],
                    'x': [-55.2, -56.2, -57.2],
                    'y': [22.1, 22.1, 22.1]})

df2 = pd.DataFrame({
    'x_coord': [-15.2],
    'y_coord': [19.1]})

euclidean = np.linalg.norm(df1[['x', 'y']].values - df2[['x_coord', 'y_coord']].values,
                           axis=1)
best_user = df1[df1.user_id == df1.user_id.max()]
# print(best_user)
#
# print(best_user[['x', 'y']].mean(axis=1).values)
print(df1.drop(
    ['user_id'],
    axis=1))
