import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


# Function to load the data from a given file path
def load_data(file_path):
    with open(file_path) as f:
        return pd.read_csv(f)


# Function to apply PCA to the input data and return the transformed data
def apply_pca(X, explained_variance_ratio_threshold=0.95):
    pca = PCA().fit(X)
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= explained_variance_ratio_threshold) + 1
    return pca.transform(X)[:, :n_components]


# Function to perform t-SNE on the PCA-transformed data
def perform_tsne(X_pca, perplexity, n_iter, init_method):
    tsne = TSNE(init=init_method, learning_rate="auto", random_state=42,
                perplexity=perplexity, n_iter=n_iter, verbose=1, n_jobs=-1)
    return tsne.fit_transform(X_pca)


# Main Function to run code
def main():
    # Load the data (features)
    df = load_data("../source_code_clustering_plots/"
                   "unmerged_mobilenetv2_image_features_no_duplicates.csv")
    X = df.iloc[:, :-1].values

    # Apply PCA to the features
    X_pca = apply_pca(X)

    # Define the features of interest
    features_of_interest = ["Magnesium (mM)"]
    data_set = load_data(
        "../../../Jordan_Connolly-nucleic-acid-origami-database/Full_Data_Set/Machine-Learning-Sets/All_Papers_Non_RevNano_Feature_Subset.csv")

    data_set_df = data_set.groupby("Paper Number").filter(lambda x: len(x) == 1)

    # Set t-SNE parameters
    perplexity = 15
    n_iter = 500
    init_method_list = ["random", "pca"]

    # Iterate through each feature of interest
    counter = 0
    for feature_of_interest in features_of_interest:
        for init_method in init_method_list:
            counter += 1
            print(f"Perplexity: {perplexity}, n_iter: {n_iter}, feature of interest: {feature_of_interest}")

            # Perform t-SNE using the current parameters and PCA-transformed data
            clusters = perform_tsne(X_pca, perplexity, n_iter, init_method)

            # Create a DataFrame from the t-SNE results
            df = pd.DataFrame(clusters, columns=["cluster1", "cluster2"])

            # Load paper numbers and add them to the DataFrame
            paper_df = load_data("../image_paper_number.csv")
            df["Paper Number"] = paper_df["0"].values.tolist()

            # Select the columns containing the paper number, experiment number, and feature of interest
            clusters = df.copy()
            selected_columns = data_set.loc[:, ['Paper Number', 'Experiment Number',
                                                'Scaffold Name', 'Scaffold Length (bases)',
                                                'Structure Dimension', 'Yield (%)', 'Yield Range (%)',
                                                'Characterised By', 'Scaffold to Staple Ratio', 'Constructed By',
                                                'Buffer Name', 'MgCl2 Used', 'Magnesium Acetate Used',
                                                'Peak Temperature (oC)', 'Base Temperature (oC)',
                                                'Scaffold Molarity (nM)',
                                                'nanostructure length (nm)', 'nanostructure width (nm)',
                                                'number of individual staples',
                                                'overall buffer pH', 'TRIS-HCl (mM)', 'Boric Acid (mM)', 'NaCl (mM)',
                                                'Acetate (mM)', 'Acetic acid (mM)', 'EDTA (mM)', 'Temperature Ramp (s)',
                                                "Staple Molarity (nM)", "Thermal Profile",
                                                feature_of_interest]]

            # Merge the selected columns with the t-SNE clusters using the paper number as a key
            merged_df = pd.merge(selected_columns, clusters, on="Paper Number")
            merged_df.sort_values(feature_of_interest, inplace=True)
            merged_df.to_csv("merged_df_stored_for_dash/" +
                             f"merged_df_for_dash_app__{perplexity}_{n_iter}.csv")

            # Return the merged_df and feature_of_interest for the last iteration
            if counter == len(features_of_interest) * len(init_method_list):
                return merged_df, feature_of_interest


if __name__ == "__main__":
    main()
