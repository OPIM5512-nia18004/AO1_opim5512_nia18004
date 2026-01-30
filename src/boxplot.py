from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
# print(df.head())
# print(df.shape)
# print(df.columns)

bins = [0, 3, 4, 5, 6, 50]
labels = ['0-3', '3-4', '4-5', '5-6', '6+']

df['Room_Range'] = pd.cut(df['AveRooms'], bins=bins, labels=labels)

House_Val_boxplot = df.boxplot(column = "MedHouseVal", by= "Room_Range", figsize = (8,4))
House_Val_boxplot.set_ylabel("Median House Value")
House_Val_boxplot.set_xlabel("Average # of Rooms (binned)")
House_Val_boxplot.set_title("Median House Value by Avg. Rooms")

#save the boxplot
plt.savefig('figs/boxplot.png')

#plt.show()


