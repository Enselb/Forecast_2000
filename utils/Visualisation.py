import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualisation( y_true, y_pred):
    plt.figure(figsize= (15,5))
    sns.lineplot(y_pred, marker = "o")
    sns.lineplot(y_true, marker = "o")
    return y_true, y_pred
