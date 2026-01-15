# %%
import polars as pl
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from lpm_plot import plot_heatmap

df = pl.read_parquet("pairwise_mutual_information.parquet")

# The following for scipy clustering method: 
# matrix = df.pivot(index="column1", columns="column2", values="score").fillna(0)
# row_dist = pdist(matrix)
# row_linkage = linkage(row_dist, method="average")
# row_order = [matrix.index[i] for i in leaves_list(row_linkage)]

# col_dist = pdist(matrix.T)
# col_linkage = linkage(col_dist, method="average")
# col_order = [matrix.columns[i] for i in leaves_list(col_linkage)]


# Problem: Cluster the data while taking into account the actual value of the "score" column
plot_heatmap(df, detailed_df=None, interactive=True)


# Column 1, Column 2, Score (Mutual Information)
# For each column, compute some additional overall metric to rank mutual information across all the other columns
# Create additional column to the matrix and then sort by that column 

# %%
