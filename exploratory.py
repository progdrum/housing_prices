import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from clean import clean_train, partial_clean_train


# The following code assumes data as cleaned up in clean.py

# Plot a simple distribution of the sale prices of houses in the training set
sns.set_style("darkgrid")
sns.distplot(clean_train['SalePrice'])

# Get a correlation for numerical features and output a heatmap of correlations
corr_frame = clean_train.corr()
sns.heatmap(pd.DataFrame(corr_frame.loc['SalePrice']))

# Output strip plots of categorical variables
for num, col in enumerate(partial_clean_train.columns):
    if partial_clean_train[col].dtype is np.dtype('object'):
        plt.figure(num)
        sns.stripplot(x=col, y='SalePrice', data=partial_clean_train, jitter=True)
