import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import pickle
import matplotlib.patches as patches


def data_importation(year=2018, month=1, day=1):
    # Constructs the path to a numpy data file based on year, month, and days
    # and loads it as an array while converting negative values to NaN.
    base_path = "C:\\Users\\fayca\\Desktop\\PROJET S8\\data"
    # Construisez le chemin complet en utilisant l'année, le mois et le jour
    full_path = os.path.join(base_path, str(year), f'RR_IDF300x300_{year:04d}{month:02d}{day:02d}.npy')
    RR = np.load(full_path) / 100.0
    RR[RR < 0] = np.nan  # Replace negative values with NaN
    return RR

def dtw_distance(x, y):
    # Calculates the Dynamic Time Warping distance between two time series x and y.
    distance, _ = fastdtw(x, y)
    return distance

def calculate_distance_matrix(RR, x_start=0, x_end=10, y_start=0, y_end=10, distance_function=dtw_distance):
    # Validates the specified indices and calculates a distance matrix for the subsection of RR
    # using the specified distance function (default is DTW).
    if x_end > 300 or y_end > 300 or x_start < 0 or y_start < 0:
        print("Index problem: Check the boundaries.")
        return None
    selected_data = RR[:, x_start:x_end, y_start:y_end]
    num_series = selected_data.shape[1] * selected_data.shape[2]
    distance_matrix = np.zeros((num_series, num_series))
    for i in range(num_series):
        for j in range(i + 1, num_series):
            x_i, y_i = divmod(i, selected_data.shape[2])
            x_j, y_j = divmod(j, selected_data.shape[2])
            ts1 = selected_data[:, x_i, y_i]
            ts2 = selected_data[:, x_j, y_j]
            distance = distance_function(ts1, ts2)
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    return distance_matrix

def DBSCAN_Slinding_Window(RR, longitude_max=2, latitude_max=2, step=9, width=10, eps=10, min_samples=2, metric=dtw_distance):
    # Applies DBSCAN clustering to data within a sliding window across the dataset RR.
    # Returns dictionaries mapping cluster information, labels, and core points indices.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dict_clusters = {}
    dict_labels = {}
    core_points = {}
    for i in range(longitude_max):
        for j in range(latitude_max):
            data = RR[:, i*step:width+i*step, j*step:width+j*step].reshape(-1, 288)
            dbscan.fit(data)
            cluster_key = f"cluster({i},{j})"
            label_key = f"labels({i},{j})"
            core_points_key = f"core_points({i},{j})"
            dict_clusters[cluster_key] = dbscan
            dict_labels[label_key] = dbscan.labels_.reshape(width, width)
            core_points[core_points_key] = dbscan.core_sample_indices_
    return dict_clusters, dict_labels, core_points

def process_pair(i, j, RR, step, width):
    distance_matrix = calculate_distance_matrix(
        RR, i * step, width + i * step, j * step, width + j * step, dtw_distance
    )
    return f"distance_matrix{i},{j}", distance_matrix

def saving_matrix(RR, step, width, longitude_max, latitude_max):
    d = {}
    with ProcessPoolExecutor() as executor:
        # Generate all pairs to be processed in parallel
        
        tasks = {(i, j): executor.submit(process_pair, i, j, RR, step, width)
                 for i in range(longitude_max) for j in range(latitude_max)}
        # Barre de progression pour suivre l'avancement
        for (i, j), future in tqdm(tasks.items(), desc='Processing pairs'):
            key, matrix = future.result()
            d[key] = matrix

    # Save the dictionary to a pickle file
    file_name = f'distance_matrices_width={width}_step={step}_longitude={longitude_max}_latitude={latitude_max}_20180201.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(d, f)


def DBSCAN_DP(longitude_max=2, latitude_max=2, step=9, width=10, eps=10, min_samples=2,d_loaded=None):
    # Loads precomputed distance matrices and applies DBSCAN clustering.
    # Returns dictionaries with clustering results, labels, and core points indices.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')

    dict_clusters = {}
    dict_labels = {}
    core_points = {}
    for i in range(longitude_max):
        for j in range(latitude_max):
            matrix_key = f"distance_matrix{i},{j}"
            if matrix_key in d_loaded:
                dbscan.fit(d_loaded[matrix_key])
                cluster_key = f"cluster({i},{j})"
                label_key = f"labels({i},{j})"
                core_points_key = f"core_points({i},{j})"
                dict_clusters[cluster_key] = dbscan
                dict_labels[label_key] = dbscan.labels_.reshape(width, width)
                core_points[core_points_key] = dbscan.core_sample_indices_
    return dict_clusters, dict_labels, core_points

def affichage_clusters(dict_labels, longitude_max, latitude_max):
    # Creates a subplot grid for displaying clustering results based on the provided labels dictionary.
    _, axs = plt.subplots(longitude_max, latitude_max)
    for i in range(longitude_max):
        for j in range(latitude_max):
            axs[i, j].imshow(dict_labels[f"labels({i},{j})"])
            axs[i, j].set_title(f'clusters({i},{j})')
            axs[i, j].axis('off')
    plt.show()
    


def affichage_clusters_enhanced(dict_labels, core_points, longitude_max, latitude_max, width):
    # Creates a subplot grid for displaying clustering results based on the provided labels dictionary.
    fig, axs = plt.subplots(longitude_max, latitude_max, figsize=(5 * latitude_max, 5 * longitude_max))
    for i in range(longitude_max):
        for j in range(latitude_max):
            label_key = f"labels({i},{j})"
            core_key = f"core_points({i},{j})"
            if label_key in dict_labels and core_key in core_points:
                axs[i, j].imshow(dict_labels[label_key], cmap='viridis')
                # Highlight core points
                for point_index in core_points[core_key]:
                    y, x = divmod(point_index, width)  # Convert index to x, y coordinates
                    circle = patches.Circle((x, y), radius=0.3, edgecolor='red', facecolor='none')
                    axs[i, j].add_patch(circle)
                axs[i, j].set_title(f'Clusters {i},{j}')
                axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()

# Example usage assuming you have dict_labels and core_points calculated
# affichage_clusters_enhanced(dict_labels, core_points, 2, 2, 10)


def traduction_core_points_map(step, width, i, j, core_points):
    # Translates core points indices to their respective positions in the subgroup and on the overall map.
    points_groupe = []
    points_map = []
    for point in core_points:
        points_groupe.append({"x_groupe": point % width, "y_groupe": point // width})
        points_map.append({"x_map": i * step + point % width, "y_map": j * step + point // width})
    return points_groupe, points_map

######################## suite a revoir

def merge_clusters_if_shared_core_point(i1, j1, i2, j2, core_points, dict, step, width, running_max_label, processed_labels):
    """
    Merges clusters between two sections of data if they share common core points. This function is used
    to ensure that clusters which are spatially or temporally close and share core points are considered as one.
    
    Args:
    i1, j1, i2, j2: Indices of the sections being compared.
    core_points: Dictionary containing core points indices.
    dict: Dictionary containing cluster labels.
    step: The step size used in the sliding window approach.
    width: The width of the window.
    running_max_label: The current maximum label used in re-labeling to ensure uniqueness across merged clusters.
    processed_labels: Set to track labels that have been processed to avoid duplication.
    
    Returns:
    running_max_label: Updated maximum label after processing this merge.
    dict: Updated dictionary with merged labels.
    """

    # Retrieve core points and labels for both sections
    core_points_1 = core_points[f"core_points({i1},{j1})"]
    core_points_2 = core_points[f"core_points({i2},{j2})"]
    labels_1 = dict[f"labels({i1},{j1})"]
    labels_2 = dict[f"labels({i2},{j2})"]

    # Translate core points to map coordinates to handle them in a unified coordinate system
    _, points_map1 = traduction_core_points_map(step, width, i1, j1, core_points_1)
    
    _, points_map2 = traduction_core_points_map(step, width, i2, j2, core_points_2)
    
    # Find common core points between the two sections
    set_core_points1 = {tuple(point.values()) for point in points_map1}
    set_core_points2 = {tuple(point.values()) for point in points_map2}
    common_core_points = set_core_points1.intersection(set_core_points2)
    

    # Relabel clusters based on shared core points
    for point in common_core_points:
        point_group1 = (point[0] - i1 * step, point[1] - j1 * step)
        point_group2 = (point[0] - i2 * step, point[1] - j2 * step)
        
        label_1 = labels_1[point_group1[0], point_group1[1]]
        label_2 = labels_2[point_group2[0], point_group2[1]]
        if label_1 != label_2 and label_1 is not None and label_2 is not None:
            
            for i in range(labels_2.shape[0]):
                for j in range(labels_2.shape[1]):
                    if labels_2[i, j] == label_2:
                        labels_2[i, j] = label_1
            processed_labels.add(label_1)
            
            
    for point in common_core_points:
        point_group1 = (point[0] - i1 * step, point[1] - j1 * step)
        point_group2 = (point[0] - i2 * step, point[1] - j2 * step)
        label_1 = labels_1[point_group1[0], point_group1[1]]
        label_2 = labels_2[point_group2[0], point_group2[1]]
        if label_1 != label_2 and label_1 is not None and label_2 is not None:
            for i in range(labels_1.shape[0]):
                for j in range(labels_1.shape[1]):
                    if labels_1[i, j] == label_1:
                        labels_1[i, j] = label_2
            processed_labels.add(label_2)

    # Relabel the remaining points in the second section to ensure they remain unique
    for point in set_core_points2 - common_core_points:
        point_group = (point[0] - i2 * step, point[1] - j2 * step)
        label_to_change = labels_2[point_group[0], point_group[1]]
        if label_to_change not in processed_labels:
            for i in range(labels_2.shape[0]):
                for j in range(labels_2.shape[1]):
                    if labels_2[i, j] == label_to_change:
                        labels_2[i, j] = running_max_label
            processed_labels.add(label_to_change)
            running_max_label += 1

    # Update the labels in the dictionary for both sections
    dict[f"labels({i1},{j1})"] = labels_1
    dict[f"labels({i2},{j2})"] = labels_2
    
    return running_max_label,processed_labels, dict

def merge_clusters_if_shared_core_point(i1, j1, i2, j2, core_points, dict, step, width, running_max_label, processed_labels, min_cluster_size=10):
    """ 
    Merges clusters between two sections of data if they share common core points. 
    Small clusters that do not meet the minimum size threshold are classified as noise.

    Args:
        i1, j1, i2, j2: Indices of the sections being compared.
        core_points: Dictionary containing core points indices.
        dict: Dictionary containing cluster labels.
        step: The step size used in the sliding window approach.
        width: The width of the window.
        running_max_label: The current maximum label used in re-labeling to ensure uniqueness across merged clusters.
        processed_labels: Set to track labels that have been processed to avoid duplication.
        min_cluster_size: Minimum number of points a cluster must have to not be considered noise.

    Returns:
        running_max_label: Updated maximum label after processing this merge.
        dict: Updated dictionary with merged labels.

    """
    # Retrieve core points and labels for both sections
    core_points_1 = core_points[f"core_points({i1},{j1})"]
    core_points_2 = core_points[f"core_points({i2},{j2})"]
    labels_1 = dict[f"labels({i1},{j1})"]
    labels_2 = dict[f"labels({i2},{j2})"]

    # Translate core points to map coordinates
    _, points_map1 = traduction_core_points_map(step, width, i1, j1, core_points_1)
    _, points_map2 = traduction_core_points_map(step, width, i2, j2, core_points_2)
    
    # Find common core points between the two sections
    set_core_points1 = {tuple(point.values()) for point in points_map1}
    set_core_points2 = {tuple(point.values()) for point in points_map2}
    common_core_points = set_core_points1.intersection(set_core_points2)
    
    # Relabel clusters based on shared core points
    for point in common_core_points:
        point_group1 = (point[0] - i1 * step, point[1] - j1 * step)
        point_group2 = (point[0] - i2 * step, point[1] - j2 * step)
        
        label_1 = labels_1[point_group1[0], point_group1[1]]
        label_2 = labels_2[point_group2[0], point_group2[1]]
        if label_1 != label_2 and label_1 is not None and label_2 is not None:
            for i in range(labels_2.shape[0]):
                for j in range(labels_2.shape[1]):
                    if labels_2[i, j] == label_2:
                        labels_2[i, j] = label_1
            processed_labels.add(label_1)

    # Treat small clusters as noise
    for point in set_core_points2 - common_core_points:
        point_group = (point[0] - i2 * step, point[1] - j2 * step)
        label_to_change = labels_2[point_group[0], point_group[1]]
        if label_to_change not in processed_labels and np.count_nonzero(labels_2 == label_to_change) < min_cluster_size:
            # Mark as noise by setting label to -1
            labels_2[labels_2 == label_to_change] = -1
            processed_labels.add(label_to_change)

    # Update the labels in the dictionary for both sections
    dict[f"labels({i1},{j1})"] = labels_1
    dict[f"labels({i2},{j2})"] = labels_2
    
    return running_max_label, processed_labels, dict


import pickle

def save_distance_matrices(RR, width, step, longitude_max, latitude_max):
    """
    Calculates and saves a dictionary of distance matrices for each sub-section of the RR data array.
    
    Parameters:
    RR : numpy.ndarray
        The data array containing time series data for each point.
    width : int
        The width of each sub-section.
    step : int
        The step size between the starts of each sub-section.
    longitude_max : int
        The maximum number of sub-sections along the longitude.
    latitude_max : int
        The maximum number of sub-sections along the latitude.
    """
    distance_matrices = {}
    for i in range(longitude_max):
        for j in range(latitude_max):
            start_x = i * step
            end_x = start_x + width
            start_y = j * step
            end_y = start_y + width
            # Ensure we do not exceed the dimensions of RR
            if end_x <= RR.shape[1] and end_y <= RR.shape[2]:
                distance_matrices[f"distance_matrix{i},{j}"] = calculate_distance_matrix(
                    RR, start_x, end_x, start_y, end_y, dtw_distance
                )

    # Save the dictionary to a file
    filename = f'distance_matrices_width={width}_step={step}_longitude={longitude_max}_latitude={latitude_max}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(distance_matrices, f)
    print(f"Saved distance matrices to {filename}")
    

def visualize_core_points(i, j, core_points, ax, width):
    """Plots core points on the given axis for a specific subplot.

    Args:
        i (int): The row index of the subplot.
        j (int): The column index of the subplot.
        core_points (dict): Dictionary of core points indices.
        ax (matplotlib axis): The axis on which to plot.
        width (int): The width of each subplot in terms of data points.

    This function visualizes each core point for the specified subplot as a red dot.
    """
    core_key = f"core_points({i},{j})"
    if core_key in core_points:
        for point_index in core_points[core_key]:
            y, x = divmod(point_index, width)  # Convert index to x, y coordinates
            ax.plot(j * width + x, i * width + y, 'ro', markersize=1)  # Adjust marker size or color as needed

            
def visualize_cluster_map(dict_labels, core_points, longitude_max, latitude_max, width, show_core_points,eps,min_samples):
    """Generates and displays a complete map of clusters with optional core points.

    Args:
        dict_labels (dict): Dictionary containing cluster labels for each section.
        core_points (dict): Dictionary containing core points indices.
        longitude_max (int): Number of subsections along the longitude.
        latitude_max (int): Number of subsections along the latitude.
        width (int): The width of each subsection in terms of data points.
        show_core_points (bool): Flag to determine whether to show core points on the map.

    Returns:
        full_map (numpy array): The full cluster map as a numpy array.

    This function aggregates all subsections into a single map and optionally overlays core points.
    """
    full_map = np.zeros((longitude_max * width, latitude_max * width))
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    for i in range(longitude_max):
        for j in range(latitude_max):
            label_key = f"labels({i},{j})"
            if label_key in dict_labels:
                section = dict_labels[label_key]
                full_map[i*width:(i+1)*width, j*width:(j+1)*width] = section
                if show_core_points:
                    visualize_core_points(i, j, core_points, ax, width)
    
    #full_map = np.rot90(full_map,-1)
    
    mappable = ax.imshow(full_map, cmap='viridis', interpolation='none')
    
    title=f"Map of Precipitation eps={eps} min_sample={min_samples}"
    if show_core_points:
        title=title+" with core points"
    plt.title(title)
    plt.axis('off')
    plt.show()
    return full_map




def generate_final_cluster_map(pickle_file, longitude_max=33, latitude_max=33, step=9, width=10, eps=10, min_samples=3, show_core_points=False,min_cluster_size=10):
    """Main function to generate and visualize a final cluster map from pre-calculated distance matrices.

    Args:
        pickle_file (str): Path to the file containing the pre-calculated distance matrices.
        longitude_max, latitude_max (int): Dimensions of the grid of subsections.
        step (int): Step size used in the sliding window when initially creating the distance matrices.
        width (int): The width of each grid subsection.
        eps, min_samples (int): DBSCAN parameters for the clustering process.
        show_core_points (bool): Whether to display core points on the final visualization.
        min_cluster_size (int): Minimum size a cluster needs to not be considered noise.

    Returns:
        A visual representation of the clusters along with any core points if specified.

    This function integrates data loading, clustering, merging, and visualization into a comprehensive workflow.
    """
    # Load pre-calculated distance matrices
    try:
        with open(pickle_file, 'rb') as f:
            d_loaded = pickle.load(f)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

    # Apply DBSCAN for each sub-section
    _, dict_labels, core_points = DBSCAN_DP(
        longitude_max=longitude_max, latitude_max=latitude_max, step=step, 
        width=width, eps=eps, min_samples=min_samples, d_loaded=d_loaded)
    
    # Merge clusters if necessary
    processed_labels = set()
    running_max_label = np.max(dict_labels.get("labels(0,0)", [0]))

    # Process merging in all directions (two passes)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up

    def process_merging(i, j, directions):
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < longitude_max and 0 <= nj < latitude_max:
                nonlocal running_max_label, processed_labels, dict_labels
                running_max_label, processed_labels, dict_labels = merge_clusters_if_shared_core_point(
                    i, j, ni, nj, core_points, dict_labels, step, width, running_max_label, processed_labels,min_cluster_size=min_cluster_size)

    # First pass: Top-Left to Bottom-Right
    for i in range(longitude_max):
        for j in range(latitude_max):
            process_merging(i, j, directions)
    

    # Second pass: Bottom-Right to Top-Left
    for i in range(longitude_max - 1, -1, -1):
        for j in range(latitude_max - 1, -1, -1):
            process_merging(i, j, directions[::-1]) # Reverse the direction for backward check
    
    # Prepare the final map for visualization
    return visualize_cluster_map(dict_labels, core_points, longitude_max, latitude_max, width, show_core_points,eps,min_samples)
