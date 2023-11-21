import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path):
    with open(file_path) as f:
        return pd.read_csv(f)


def apply_pca(X, explained_variance_ratio_threshold=0.60):
    pca = PCA().fit(X)
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= explained_variance_ratio_threshold) + 1
    return pca.transform(X)[:, :n_components]


def perform_tsne(X_pca, perplexity, n_iter, init_method):
    tsne = TSNE(init=init_method, learning_rate="auto", random_state=42,
                perplexity=perplexity, n_iter=n_iter, verbose=1, n_jobs=-1)
    return tsne.fit_transform(X_pca)


def plot_tsne_results(merged_df, feature_of_interest, counter, init_method, perplexity, n_iter, palette):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=merged_df, x="cluster1", y="cluster2", hue=feature_of_interest, legend='full', palette=palette)
    plt.legend(title=feature_of_interest, bbox_to_anchor=(0, 0), loc=3)
    plt.title(f"AFM Image Features clustered using TSNE with {feature_of_interest} hue")
    plt.tight_layout()
    output_dir = 'tsne_image_clusters_solo_experiment_papers_60_percent'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir,
                             f"{counter}_{init_method}_perp_{perplexity}_iter_{n_iter}_"
                             f"auto_seed42_w_{feature_of_interest}_{palette}.png"),
                bbox_inches='tight')


def main():
    plt.style.use('ggplot')

    df = load_data("unmerged_mobilenetv2_image_features_no_duplicates.csv")
    X = df.iloc[:, :-1].values

    X_pca = apply_pca(X)

    features_of_interest = ["Thermal Profile"]

    data_set = load_data(
        "../Jordan_Connolly-nucleic-acid-origami-database/Full_Data_Set/Machine-Learning-Sets/"
        "All_Papers_Non_RevNano_Feature_Subset.csv")
    data_set_df = data_set.groupby("Paper Number").filter(lambda x: len(x) == 1)
    result = data_set_df.merge(df, on="Paper Number")

    perplexity = 15
    n_iter = 500
    init_method_list = ["random", "pca"]

    counter = 0
    for feature_of_interest in features_of_interest:
        for init_method in init_method_list:
            counter += 1
            print(f"Perplexity: {perplexity}, n_iter: {n_iter}, feature of interest: {feature_of_interest}")

            clusters = perform_tsne(X_pca, perplexity, n_iter, init_method)
            df = pd.DataFrame(clusters, columns=["cluster1", "cluster2"])

            paper_df = load_data("image_paper_number.csv")
            df["Paper Number"] = paper_df["0"].values.tolist()

            clusters = df.copy()
            selected_columns = data_set.loc[:, ['Paper Number', 'Experiment Number', feature_of_interest]]

            merged_df = pd.merge(selected_columns, clusters, on="Paper Number")
            merged_df.sort_values(feature_of_interest, inplace=True)

            palettes = ["pastel", "coolwarm", "husl", "Set1", "viridis"]
            for palette in palettes:
                plot_tsne_results(merged_df, feature_of_interest, counter, init_method, perplexity, n_iter, palette)


if __name__ == "__main__":
    main()
