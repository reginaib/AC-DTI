import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_heatmap(data, metric, ax):
    # Pivot the sorted DataFrame to create a matrix suitable for a heatmap
    heatmap_data = data.pivot(index="threshold_similarity", columns="threshold_affinity", values=metric)

    label = 'RMSE micro' if metric == 'rmse_micro' else 'RMSE macro'

    # Plotting the heatmap with the updated sorting
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='viridis',
                     ax=ax,
                     cbar_kws={'label': label}, linewidths=.5,
                     annot_kws={"size": 18})

    ax.set_title(f'{label}', size=18)
    ax.set_xlabel('Threshold Affinity', size=16)
    ax.set_ylabel('Threshold Similarity', size=16)
    ax.invert_yaxis()

    # Increase the size of the tick labels
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Increase the size of the color bar label
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(16)


def get_heatmap(data, metric, model_name=None, save_fig=False):
    if metric == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
        plot_heatmap(data,'rmse_micro', ax1)
        plot_heatmap(data, 'rmse_macro', ax2)
        if save_fig:
            plt.savefig(f'../analysis/{model_name}_both_heatmap.png')
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_heatmap(data, metric, ax)
        if save_fig:
            plt.savefig(f'../analysis/{model_name}_{metric}_heatmap.png')
    plt.show()


def get_pairs_heatmap(data, model_name, save_fig=False):
    # Pivot the data for the number of pairs
    heatmap_pairs = data.pivot(index="threshold_similarity", columns="threshold_affinity", values="number_of_pairs")

    heatmap_pairs = heatmap_pairs.fillna(0).astype(int)

    # Plotting the heatmap for the number of pairs
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_pairs, annot=True, fmt="d", cmap='viridis',
                     cbar_kws={'label': 'Number of Pairs'}, linewidths=.5,
                     annot_kws={"size": 14})

    plt.title('Number of Pairs per Threshold Combination',  size=18)
    ax.set_xlabel('Threshold Affinity', size=16)
    ax.set_ylabel('Threshold Similarity', size=16)
    plt.gca().invert_yaxis()

    # Increase the size of the tick labels
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Increase the size of the color bar label
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(16)


def plot_diff_heatmap(data_model1, data_model2, metric, ax):
    # Ensure the data is sorted consistently
    data_model1 = data_model1.sort_values(by=["threshold_similarity", "threshold_affinity"])
    data_model2 = data_model2.sort_values(by=["threshold_similarity", "threshold_affinity"])

    # Pivot the DataFrames to create matrices
    heatmap_data_model1 = data_model1.pivot(index="threshold_similarity", columns="threshold_affinity",
                                            values=metric)
    heatmap_data_model2 = data_model2.pivot(index="threshold_similarity", columns="threshold_affinity",
                                            values=metric)

    # Compute the difference
    heatmap_data_diff = heatmap_data_model1 - heatmap_data_model2

    label = 'RMSE micro' if metric == 'rmse_micro' else 'RMSE macro'

    # Plotting the differential heatmap
    sns.heatmap(heatmap_data_diff, annot=True, fmt=".2f", cmap='viridis',
                ax=ax,
                cbar_kws={'label': f'Difference in {label}'}, linewidths=.5,
                annot_kws={"size": 18})

    ax.set_title(f'Difference in {label} \n Higher better', size=18)
    ax.set_xlabel('Threshold Affinity', size=16)
    ax.set_ylabel('Threshold Similarity', size=16)
    ax.invert_yaxis()

    # Increase the size of the tick labels
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Increase the size of the color bar label
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(16)


def get_differential_heatmap(data_model1, data_model2, metric, title_suffix=None, save_fig=False):
    if metric == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
        plot_diff_heatmap(data_model1, data_model2, 'rmse_micro', ax1)
        plot_diff_heatmap(data_model1, data_model2, 'rmse_macro', ax2)
        if save_fig:
            plt.savefig(f'../analysis/{title_suffix}_both_diff_heatmap.png')
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_diff_heatmap(data_model1, data_model2, metric, ax)
        if save_fig:
            plt.savefig(f'../analysis/{title_suffix}_{metric}_diff_heatmap.png')

    plt.show()


def get_combined_mean_variance_heatmap(data_models1, data_models2, metric, title_suffix, save_fig=False):
    difference_accumulator = []

    for data_model1, data_model2 in zip(data_models1, data_models2):
        data_model1_sorted = data_model1.sort_values(by=["threshold_similarity", "threshold_affinity"])
        data_model2_sorted = data_model2.sort_values(by=["threshold_similarity", "threshold_affinity"])

        heatmap_data_model1 = data_model1_sorted.pivot(index="threshold_similarity", columns="threshold_affinity", values=metric)
        heatmap_data_model2 = data_model2_sorted.pivot(index="threshold_similarity", columns="threshold_affinity", values=metric)

        heatmap_data_diff = heatmap_data_model1 - heatmap_data_model2
        difference_accumulator.append(heatmap_data_diff)

    differences_df = pd.concat(difference_accumulator, axis=0)
    group_levels = differences_df.index.names

    mean_difference = differences_df.groupby(level=group_levels).mean()
    variance_difference = differences_df.groupby(level=group_levels).var()

    combined_annotation = mean_difference.map(lambda x: f"{x:.4f}") + "\n± " + variance_difference.map(lambda x: f"{x:.4f}")

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(mean_difference, annot=combined_annotation.to_numpy(), fmt="",  cmap='viridis',
                     cbar_kws={'label': f'Mean and Variance of Difference in {metric}'}, linewidths=.5, annot_kws={"size": 10})
    plt.title(f'Combined Mean and Variance Differential - {title_suffix}')
    ax.set_xlabel('Threshold Affinity', size=16)
    ax.set_ylabel('Threshold Similarity', size=16)
    plt.gca().invert_yaxis()

    if save_fig:
        plt.savefig(f'../analysis/{title_suffix}_{metric}_combined_heatmap.png')
    plt.show()


def get_hist_prop(data, col):
    # Check if the DataFrame has hierarchical indexes or multi-level columns
    if isinstance(data.index, pd.MultiIndex) or isinstance(data.columns, pd.MultiIndex):
        data = data.reset_index()

    plt.figure(figsize=(10, 6))
    plt.hist(data[col].to_numpy(), color='blue', alpha=0.5, bins=50)
    plt.xlabel(col)
    plt.ylabel('Pairs')
    plt.show()
