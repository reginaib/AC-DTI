import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_2d_plot(data, metric, threshold):
    plt.figure(figsize=(12, 12))
    sns.barplot(x=threshold, y=metric, data=data)
    plt.xticks(rotation=45)
    plt.xlabel(threshold, fontsize=16)
    plt.ylabel(metric, fontsize=16)
    plt.show()
    #plt.savefig('../analysis/morgan_cnn_kiba_predictions_RMSE.png')


def get_3d_plot(data, metric, general_performance, model_name):
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Set up the grid in the 3D plot
    xpos, ypos = np.meshgrid(data['threshold_similarity'].unique(),
                             data['threshold_affinity'].unique(), indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    # The size of each bar
    dx = dy = 0.05
    dz = data[metric].values

    # Add the general performance
    xpos = np.append(xpos, 0)
    ypos = np.append(ypos, 0)
    zpos = np.append(zpos, 0)
    dz = np.append(dz, general_performance)

    # Set the colors based on the z position
    colors = plt.cm.viridis(dz / max(dz))

    # Create the 3D bar plot
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    # Set labels
    ax.set_xlabel('Threshold similarity', fontsize=16)
    ax.set_ylabel('Threshold affinity', fontsize=16)
    ax.set_zlabel(metric, fontsize=16)

    plt.savefig(f'../analysis/{model_name}_predictions_{metric}.png')
    plt.show()
