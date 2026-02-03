# Import all the libraries we need
import pandas as pd # For data handling/"ingestion"
import numpy as np # For scientific numerical ops
import matplotlib.pyplot as plt # For plotting

# For embeddings/training
from sentence_transformers import SentenceTransformer

# For mapping and clustering
from sklearn.cluster import KMeans # widely used for cluster analysis, paritioning a dataset into similar groups based on distance.
import umap # Visualizing data in lower dimensionn (2D)

# I am using a mock dataset so lets load that into the system
df = pd.read_excel("financial_dataset_200_companies_expanded_market_cap.xlsx")
df.head()
"""Each row is a company and the colums include values/text pertaining to
sector, ticker, volatility, etc."""

""" At this point, we just have a regular spreadsheet. No map, no visualization.
All the texts and numbers are jumbled together."""
print(f"Dataset size: {df.shape}")
print(df.columns.tolist())

# The output should be the size of our dataset (how many rows - xval, how
# many columns - yval). Then, the output under should consist of the names of the
# columns per the df.columns.tolist()

def buildtext(row): # Sample description. We can access the data for a certain company with this funciton
  return f"""
  Company description: {row['business_description']} |
  Risk Level: {row['risk_category']} |
  Sector: {row['sector']} | Ticker: {row['ticker']} |
  Origin: {row['country']} | Market Cap (USD): {row['market_cap_usd']}
  """
# Here we turn each row into a summary, or small piece of text, to identify the
# characteristics of the specific company.
df["embedding_text"] = df.apply(buildtext, axis =1) # Column 1 data
df["embedding_text"].iloc[0] # Data for the first cell in column 1 in our xlsx (view spreadsheet/xlsx to confirm)

model = SentenceTransformer("all-MiniLM-L6-v2") # pre-trained model

# Each company becomes a list of numbers. Similar numbers = similar companies
# Different companies = different numbers
embeddings = model.encode(
    df["embedding_text"].tolist(), # Column 1 as defined above
    show_progress_bar=True
)
embeddings.shape

# Dimensionally reduce size of data to fit a map
reducer = umap.UMAP(
    n_neighbors = 15, # Defines the size of some neighborhoods (in this case "clusters")
    min_dist = 0.1, # Separation/distance between points
    random_state=42 # Random numbers - shuffling data before putting it on the map
)
# Move to a 2D layout
coordinates = reducer.fit_transform(embeddings)
df["x"] = coordinates[:, 0]
df["y"] = coordinates[:, 1]

# Layout of plot
plt.figure(figsize=(8,6))
plt.scatter(df["x"], df["y"], s=12, alpha=0.7) # Scattering points
plt.title("Semantic Map of Finance Companies - IAP Test")
plt.xlabel("Umap1")
plt.ylabel("Umap2")
plt.show()

# Examples via data within certain companies
plt.figure(figsize=(8,6))
plt.scatter(
    df["x"],
    df["y"],
    c=df["beta_or_volatility"],
    cmap="viridis", # For coloring
    s=15,
)
plt.colorbar(label="Volatility")
plt.title("Semantic Map of Volatility - Coloration and Overlapping IAP Test")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(
    df["x"], df["y"],
    c=np.log10(df["market_cap_usd"]), # Scaling values for better visualization
    cmap="plasma", # For coloring
    s=15,
)
plt.colorbar(label="log10(Market Cap)")
plt.title("Semantic Map of Market Cap - Coloration and Overlapping")
plt.show()

# Based on similarity
kmeans = KMeans(n_clusters =8, random_state=42) # Number of clusters and more shuffling
df["cluster"] = kmeans.fit_predict(embeddings) # Clustering

plt.figure(figsize=(8,6))
plt.scatter(
    df["x"], df["y"],
    c=df["cluster"],
    cmap="tab10", # For coloring - clusters
    s=15
)
plt.title("Semantic Map w/ Clusters")
plt.show()

# Exploring a cluster
def sample(cluster_id, n=5): #(what cluster you want to explore, how many rows of data to return)
  return df[df["cluster"] == cluster_id][ # Based on the "cluster" declaration above.
      ["ticker", "sector", "business_description", "risk_category", "market_cap_usd"]
  ].head(n) # Return data based on 5 cells/rows from the top

sample(3) # Sample cluster somewhere on the map - in this case, cluster 3

# grouping clusters based on data and average numbers per 300 companies (mean)
df.groupby("cluster") [[ # All rows belonging to the same cluster are grouped together
    "beta_or_volatility", # Data 1
    "revenue_growth_yoy", # Data 2
    "market_cap_usd" # Data 3
]].mean() # Calculating average among all values of data from all companies after grouping

# Specific clusters for comparison
cluster_a = df[df["cluster"] == 0] # All data that belongs to cluster 0 is labeled cluster_a
cluster_b = df[df["cluster"] == 1] # All data that belongs to cluster 1 is labeled cluster_b

comparison = pd.DataFrame({ # Holding the results of the comparison between the values within
    "Cluster 0 - compare with 1": cluster_a[["market_cap_usd", "revenue_growth_yoy"]].mean(),
    "Cluster 1 - compare with 0": cluster_b[["market_cap_usd", "revenue_growth_yoy"]].mean()
}) # Again average with data. Numerical data only. Text data will not work as the embeddings only respond to numerical data.
comparison # Call

selected = df[df["cluster"] == 5] # the selected is the fifth cluster, number can change based on what you want to explore
selected[["ticker", "sector"]].head() # rows from the top
# Summarizer
cluster_text = "".join(
    selected["embedding_text"].tolist()[:10]
) # Brings tofether a summary for a couple entries (10). Converting into python list form
print(cluster_text[:1000]) # Prints the first 1000 characters of the cluster_text. Showing a portion of the output without overloading info
