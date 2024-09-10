import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors


def get_pairs_number(data):
    """
    Convert the data into a pivot table, summarizing the number of pairs for each combination
    of threshold similarity and threshold affinity.

    Args:
        data (DataFrame): The input data containing columns for threshold similarity, threshold affinity,
                          and number of pairs.

    Returns:
        DataFrame: A pivot table with threshold similarity as rows, threshold affinity as columns,
                   and number of pairs as values.
    """
    heatmap_pairs = data.pivot(index="threshold_similarity",
                               columns="threshold_affinity",
                               values="number_of_pairs")
    heatmap_pairs = heatmap_pairs.fillna(0).astype(int)
    return heatmap_pairs


def create_mask(data, ax):
    """
    Create and overlay a mask on the heatmap where the number of pairs is below 100.
    This mask highlights these areas in grey.

    Args:
        data (DataFrame): The input data containing the number of pairs.
        ax (Axes): The matplotlib axes to overlay the mask on.
    """
    pairs_data = get_pairs_number(data)

    # Create a mask where the number of pairs is below 100
    mask = pairs_data < 100

    # Overlay the mask with transparency
    mask_color = mcolors.to_rgba('grey', alpha=0.5)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask.iloc[i, j]:
                # Add a rectangular patch on the cells that meet the mask condition
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                           color=mask_color, linewidth=0))


def plot_heatmap(data, metric, ax):
    """
    Plot a heatmap for the specified metric (RMSE micro or RMSE macro) based on the threshold
    similarity and threshold affinity.

    Args:
       data (DataFrame): The input data containing metrics and thresholds.
       metric (str): The metric to plot ('rmse_micro' or 'rmse_macro').
       ax (Axes): The matplotlib axes to plot the heatmap on.
    """
    # Pivot the sorted DataFrame to create a matrix suitable for a heatmap
    heatmap_data = data.pivot(index="threshold_similarity",
                              columns="threshold_affinity",
                              values=metric)

    label = 'RMSE micro' if metric == 'rmse_micro' else 'RMSE macro'

    # Plotting the heatmap with the updated sorting
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='viridis',
                ax=ax,
                cbar_kws={'label': label}, linewidths=.5,
                annot_kws={"size": 20})

    # Overlay a mask on the heatmap
    create_mask(data, ax)

    ax.set_title(f'{label}', size=24)
    ax.set_xlabel('Threshold Affinity', size=26)
    ax.set_ylabel('Threshold Similarity', size=26)
    ax.invert_yaxis()

    # Increase the size of the tick labels
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    # Customize the color bar label size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(22)
    cbar.ax.tick_params(labelsize=20)


def get_heatmap(data, metric, model_name=None, save_fig=False):
    """
    Generate and display a heatmap for the specified metric.
    Optionally, generate heatmaps for both RMSE micro and macro.

    Args:
        data (DataFrame): The input data containing metrics and thresholds.
        metric (str): The metric to plot ('rmse_micro', 'rmse_macro', or 'both').
        model_name (str, optional): The model name to use when saving the figure. Defaults to None.
        save_fig (bool, optional): Whether to save the heatmap as a file. Defaults to False.
    """
    if metric == 'both':
        # Plot heatmaps for both RMSE micro and RMSE macro side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
        plot_heatmap(data, 'rmse_micro', ax1)
        plot_heatmap(data, 'rmse_macro', ax2)

        if save_fig:
            plt.savefig(f'../analysis/{model_name}_both_heatmap.pdf', dpi=300, bbox_inches='tight')
    else:
        # Plot a single heatmap for the specified metric
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_heatmap(data, metric, ax)
        if save_fig:
            plt.savefig(f'../analysis/figs/{model_name}_{metric}_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def get_pairs_heatmap(data, model_name, save_fig=False):
    """
    Generate and display a heatmap showing the number of pairs for each combination of threshold similarity
    and threshold affinity.

    Args:
        data (DataFrame): The input data containing number of pairs and thresholds.
        model_name (str): The model name to use when saving the figure.
        save_fig (bool, optional): Whether to save the heatmap as a file. Defaults to False.
    """

    # Pivot the data for the number of pairs
    heatmap_pairs = get_pairs_number(data)

    # Plotting the heatmap for the number of pairs
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_pairs,
                     annot=True, fmt="d", cmap='viridis',
                     cbar_kws={'label': 'Number of Pairs'}, linewidths=.5,
                     annot_kws={"size": 12})

    # Overlay a mask on the heatmap
    create_mask(data, ax)

    plt.title('Number of Pairs per Threshold Combination', size=22)
    ax.set_xlabel('Threshold Affinity', size=20)
    ax.set_ylabel('Threshold Similarity', size=20)
    plt.gca().invert_yaxis()

    # Increase the size of the tick labels
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # Increase the size of the color bar label
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(20)
    cbar.ax.tick_params(labelsize=20)

    if save_fig:
        plt.savefig(f'../analysis/figs/{model_name}_pair_number_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_mean_sd_heatmap(data, metric, ax):
    metric_accumulator = []

    for data_model1 in data:
        # Extract the metric for each model and pivot it into a heatmap format
        heatmap_data = data_model1.pivot(index="threshold_similarity",
                                         columns="threshold_affinity",
                                         values=metric)
        metric_accumulator.append(heatmap_data)

    label = 'RMSE micro' if metric == 'rmse_micro' else 'RMSE macro'
    # Concatenate the metric DataFrames and compute the mean and standard deviation
    concatenated_df = pd.concat(metric_accumulator, axis=0)
    group_levels = concatenated_df.index.names

    mean_metric = concatenated_df.groupby(level=group_levels).mean()
    sd_metric = concatenated_df.groupby(level=group_levels).std()

    # Combine the mean and standard deviation into a single annotation
    combined_annotation = (mean_metric.map(lambda x: f"{x:.3f}") + "\n±\n "
                           + sd_metric.map(lambda x: f"{x:.3f}"))

    # Plot the heatmap for the mean and standard deviation
    sns.heatmap(mean_metric, annot=combined_annotation.to_numpy(), fmt="", cmap='viridis', ax=ax,
                cbar_kws={'label': f'{label}'}, linewidths=.5,
                annot_kws={"size": 14})

    # Overlay a mask on the heatmap
    for data_model1 in data:
        create_mask(data_model1, ax)

    ax.set_title(f'Mean and Standard Deviation for {label}', size=24)
    ax.set_xlabel('Threshold Affinity', size=26)
    ax.set_ylabel('Threshold Similarity', size=26)
    ax.invert_yaxis()

    # Increase the size of the tick labels
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    # Increase the size of the color bar label
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(22)
    cbar.ax.tick_params(labelsize=20)


def get_mean_sd_heatmap(data, metric, model_name=None, save_fig=False):
    if metric == 'both':
        # Plot heatmaps for both RMSE micro and RMSE macro side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
        plot_mean_sd_heatmap(data, 'rmse_micro', ax1)
        plot_mean_sd_heatmap(data, 'rmse_macro', ax2)

        if save_fig:
            plt.savefig(f'../analysis/figs/{model_name}_both_heatmap.pdf', dpi=300, bbox_inches='tight')
    else:
        # Plot a single heatmap for the specified metric
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_mean_sd_heatmap(data, metric, ax)
        if save_fig:
            plt.savefig(f'../analysis/figs/{model_name}_{metric}_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def compute_diff(data_model1, data_model2, metric):
    """
    Compute the difference between two models for a given metric.

    Args:
        data_model1 (DataFrame): The first model's data.
        data_model2 (DataFrame): The second model's data.
        metric (str): The metric to compare (e.g., 'rmse_micro', 'rmse_macro').

    Returns:
        DataFrame: The difference between the two models for the specified metric.
    """

    # Ensure the data is sorted consistently
    data_model1 = data_model1.sort_values(by=["threshold_similarity", "threshold_affinity"])
    data_model2 = data_model2.sort_values(by=["threshold_similarity", "threshold_affinity"])

    # Pivot the DataFrames to create matrices
    heatmap_data_model1 = data_model1.pivot(index="threshold_similarity",
                                            columns="threshold_affinity",
                                            values=metric)
    heatmap_data_model2 = data_model2.pivot(index="threshold_similarity",
                                            columns="threshold_affinity",
                                            values=metric)

    # Compute the difference between the two models
    heatmap_data_diff = heatmap_data_model1 - heatmap_data_model2

    return heatmap_data_diff


def plot_diff_heatmap(data_model1, data_model2, metric, ax):
    """
    Plot a heatmap showing the difference between two models for a given metric.

    Args:
        data_model1 (DataFrame): The first model's data.
        data_model2 (DataFrame): The second model's data.
        metric (str): The metric to compare (e.g., 'rmse_micro', 'rmse_macro').
        ax (Axes): The matplotlib axes to plot the heatmap on.
    """
    # Compute the difference between the two models
    heatmap_data_diff = compute_diff(data_model1, data_model2, metric)

    label = 'RMSE micro' if metric == 'rmse_micro' else 'RMSE macro'

    # Plotting the differential heatmap
    sns.heatmap(heatmap_data_diff, annot=True, fmt=".2f", cmap='viridis', ax=ax,
                cbar_kws={'label': f'Difference in {label}'}, linewidths=.5,
                annot_kws={"size": 20})

    # Overlay a mask on the heatmap
    create_mask(data_model1, ax)

    ax.set_title(f'Difference in {label} \n Higher better', size=24)
    ax.set_xlabel('Threshold Affinity', size=26)
    ax.set_ylabel('Threshold Similarity', size=26)
    ax.invert_yaxis()

    # Increase the size of the tick labels
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    # Increase the size of the color bar label
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(22)
    cbar.ax.tick_params(labelsize=20)


def get_differential_heatmap(data_model1, data_model2, metric, title_suffix=None, save_fig=False):
    """
    Generate and display a differential heatmap comparing two models for a given metric.

    Args:
        data_model1 (DataFrame): The first model's data.
        data_model2 (DataFrame): The second model's data.
        metric (str): The metric to compare (e.g., 'rmse_micro', 'rmse_macro', or 'both').
        title_suffix (str, optional): Suffix to use when saving the figure. Defaults to None.
        save_fig (bool, optional): Whether to save the heatmap as a file. Defaults to False.
    """
    if metric == 'both':
        # Plot differential heatmaps for both RMSE micro and RMSE macro side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

        plot_diff_heatmap(data_model1, data_model2, 'rmse_micro', ax1)
        plot_diff_heatmap(data_model1, data_model2, 'rmse_macro', ax2)
        if save_fig:
            plt.savefig(f'../analysis/figs/{title_suffix}_both_diff_heatmap.pdf',
                        dpi=300, bbox_inches='tight')
    else:
        # Plot a single differential heatmap for the specified metric
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_diff_heatmap(data_model1, data_model2, metric, ax)
        if save_fig:
            plt.savefig(f'../analysis/figs/{title_suffix}_{metric}_diff_heatmap.pdf',
                        dpi=300, bbox_inches='tight')

    plt.show()


def plot_mean_sd_diff_heatmap(data_models1, data_models2, metric, ax):

    difference_accumulator = []

    for data_model1, data_model2 in zip(data_models1, data_models2):
        # Compute the difference for each pair of models
        heatmap_data_diff = compute_diff(data_model1, data_model2, metric)
        difference_accumulator.append(heatmap_data_diff)

    label = 'RMSE micro' if metric == 'rmse_micro' else 'RMSE macro'

    # Concatenate the differences and compute the mean and standard deviation
    differences_df = pd.concat(difference_accumulator, axis=0)
    group_levels = differences_df.index.names

    mean_difference = differences_df.groupby(level=group_levels).mean()
    sd_difference = differences_df.groupby(level=group_levels).std()

    # Combine the mean and standard deviation into a single annotation
    combined_annotation = (mean_difference.map(lambda x: f"{x:.3f}") + "\n±\n"
                           + sd_difference.map(lambda x: f"{x:.3f}"))

    # Plot the heatmap for the combined mean and standard deviation
    ax = sns.heatmap(mean_difference, annot=combined_annotation.to_numpy(), fmt="", cmap='viridis', ax=ax,
                     cbar_kws={'label': f'Difference in {label}'}, linewidths=.5,
                     annot_kws={"size": 14})

    # Overlay a mask on the heatmap
    for data_model1 in data_models1:
        create_mask(data_model1, ax)

    ax.set_title(f'Differential {label} \n Higher better', size=24)
    ax.set_xlabel('Threshold Affinity', size=26)
    ax.set_ylabel('Threshold Similarity', size=26)
    ax.invert_yaxis()

    # Increase the size of the tick labels
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    # Increase the size of the color bar label
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(22)
    cbar.ax.tick_params(labelsize=20)


def get_mean_sd_diff_heatmap(data_models1, data_models2, metric, title_suffix=None, save_fig=False):

    if metric == 'both':
        # Plot differential heatmaps for both RMSE micro and RMSE macro side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

        plot_mean_sd_diff_heatmap(data_models1, data_models2, 'rmse_micro', ax1)
        plot_mean_sd_diff_heatmap(data_models1, data_models2, 'rmse_macro', ax2)
        if save_fig:
            plt.savefig(f'../analysis/figs/{title_suffix}_both_diff_heatmap.pdf',
                        dpi=300, bbox_inches='tight')
    else:
        # Plot a single differential heatmap for the specified metric
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_mean_sd_diff_heatmap(data_models1, data_models2, metric, ax)
        if save_fig:
            plt.savefig(f'../analysis/figs/{title_suffix}_{metric}_diff_heatmap.pdf',
                        dpi=300, bbox_inches='tight')

    plt.show()


