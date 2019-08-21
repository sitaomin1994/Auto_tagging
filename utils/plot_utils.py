import pandas as pd
import numpy as np
from ast import literal_eval
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


# pie plot
def pie_plot(df, column_name):
    '''
    plot pie chart
    '''
    labels = list(df[column_name].value_counts().index)
    value = list(df[column_name].value_counts().values)

    fig = plt.figure(figsize=(20,15))
    plt.pie(value, labels=labels,  autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()