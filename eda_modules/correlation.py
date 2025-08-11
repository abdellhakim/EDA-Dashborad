import seaborn as sns
import matplotlib.pyplot as plt

def generate_correlation(df):
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    return fig
