import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import imageio
import contextily as cx

N_CLUSTERS = 10
IMAGE_DIR = 'images'
TMP_DIR = 'tmp'
TIME_STEPS_PER_DAY = 288


def prepare_directories():
    for directory in [IMAGE_DIR, TMP_DIR]:
        os.makedirs(directory, exist_ok=True)


def sample_data(df):
    return df.sample(100000)


def plot_pairwise_relationships(df_sample):
    features = ['duration', 'max_intensity',
                'mean_intensity', 'variance', 'percentage_null']
    sns.pairplot(df_sample, diag_kind="kde")
    sns.pairplot(df_sample, hue="label", vars=features, palette='tab10')


def plot_kde_distributions(df_sample):
    attributes = [
        ("duration", (0, 100)),
        ("max_intensity", (0, 20)),
        ("mean_intensity", (0, 6)),
        ("variance", (0, 10)),
        ("percentage_null", (0, 0.5))
    ]
    for attr, clip in attributes:
        sns.kdeplot(data=df_sample, x=attr, hue='label', fill=True,
                    common_norm=False, common_grid=True, palette='tab10', clip=clip)
        plt.show()


def get_first_day_events(df):
    return df[df['day'] == 1]


def plot_background():
    west, south, east, north = (
        0.2986403353389431, 47.502048648648646, 4.398959664661057, 50.204751351351355)
    paris_img, paris_ext = cx.bounds2img(
        west, south, east, north, ll=True, source=cx.providers.CartoDB.Voyager)
    f, ax = plt.subplots(1, figsize=(9, 9))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(f'{IMAGE_DIR}/background.png',
                bbox_inches='tight', pad_inches=0)


def create_legend(labels_unique):
    legend_elements = [mlines.Line2D([], [], marker='o', color=sns.color_palette('Set2', n_colors=N_CLUSTERS)[
                                     i], label=str(label), markersize=10, linestyle='None') for i, label in enumerate(labels_unique)]
    legend_fig, legend_ax = plt.subplots(figsize=(4, 2))
    legend_ax.legend(handles=legend_elements, loc='center')
    legend_ax.axis('off')
    legend_fig.savefig(f'{IMAGE_DIR}/legend.png',
                       bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(legend_fig)


def animate_events(events_sample, attribute='label'):
    prepare_directories()  # Ensure required directories exist
    plot_background()  # Plot and save the background image
    legend_img = plt.imread(f'{IMAGE_DIR}/legend.png')
    background_img = plt.imread(f'{IMAGE_DIR}/background.png')

    x_min, x_max = events_sample['i'].min(), events_sample['i'].max()
    y_min, y_max = events_sample['j'].min(), events_sample['j'].max()
    padding = 0.001  # Adjust padding if necessary

    for t in range(TIME_STEPS_PER_DAY):
        current_events = events_sample[(events_sample['start_time_absolute'] <= t) & (
            t <= events_sample['end_time_absolute'])]
        plt.figure(figsize=(10, 10))
        plt.gca().set_aspect('equal')
        sns.scatterplot(data=current_events, x='i', y='j', hue=attribute,
                        alpha=0.8, s=3, palette='tab10', legend=False)

        plt.xlim(x_min - (x_max - x_min) * padding,
                 x_max + (x_max - x_min) * padding)
        plt.ylim(y_min - (y_max - y_min) * padding,
                 y_max + (y_max - y_min) * padding)
        plt.imshow(background_img, aspect='equal', extent=[
                   x_min, x_max, y_min, y_max], zorder=0)
        plt.figimage(legend_img, xo=460, yo=660, zorder=9, alpha=1)

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(f'{TMP_DIR}/frame_{t:03d}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free up memory and start fresh for the next frame

    # Combine all frames into a GIF
    frames = sorted([f for f in os.listdir(TMP_DIR) if f.startswith('frame_') and f.endswith('.png')],
                    key=lambda x: int(x.split('_')[1].split('.')[0]))
    images = [imageio.imread(f"{TMP_DIR}/{frame}") for frame in frames]
    imageio.mimsave(f'{IMAGE_DIR}/animated_events.gif', images, duration=0.5)


def main():
    df = pd.read_csv('january_cluster.csv')  # Replace with your data file path
    df_sample = sample_data(df)

    # Pairwise relationships and KDE plots
    plot_pairwise_relationships(df_sample)
    plot_kde_distributions(df_sample)

    # Prepare for animations
    events_sample = get_first_day_events(df)
    create_legend(events_sample['label'].unique())

    # Animate and create GIF
    animate_events(events_sample)


if __name__ == "__main__":
    main()
