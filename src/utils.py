from joblib import Parallel, delayed
import numpy as np
import time
from typing import List, Tuple

GRID_SIZE = 300
TIME_STEPS_PER_DAY = 288


def read_data(year: int, month: int, day: int, threshold: float = 1e-3) -> np.ndarray:
    path = f"data/raw_data/{year:04d}/"
    file_name = f"RR_IDF300x300_{year:04d}{month:02d}{day:02d}.npy"
    full_path = path + file_name
    try:
        raw_data = np.load(full_path) / 100.0
        raw_data[raw_data < 0] = np.nan
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {full_path} does not exist.")
    raw_data[raw_data < threshold] = 0
    return raw_data


def segment_events(time_series: np.ndarray) -> List[Tuple[int, int]]:
    events = []
    current_event_start = None
    last_non_zero_index = -1

    for i in range(len(time_series)):
        current_value = time_series[i]
        if current_value != 0 and not np.isnan(current_value):
            if current_event_start is None:
                current_event_start = i
            elif i > 0 and (i - last_non_zero_index) > 5:
                events.append((current_event_start, last_non_zero_index))
                current_event_start = i
        else:
            if current_event_start is not None and (i - last_non_zero_index) > 5:
                events.append((current_event_start, last_non_zero_index))
                current_event_start = None

        if current_value != 0 and not np.isnan(current_value):
            last_non_zero_index = i

    if current_event_start is not None:
        events.append((current_event_start, last_non_zero_index))

    return events


def get_events(raw_data: np.ndarray) -> np.ndarray:
    events = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pixel_events = segment_events(raw_data[:, i, j])
            events[i, j] = pixel_events
    return events


def timer(func):
    """
    Decorator that prints the execution time of the function it decorates.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {
              end_time - start_time:.4f} seconds")
        return result

    return wrapper


def segmentation_events_legacy(year: int, month: int, day: int) -> Tuple[np.ndarray, np.ndarray]:
    raw_data = read_data(year, month, day)
    raw_events = np.zeros((TIME_STEPS_PER_DAY, GRID_SIZE, GRID_SIZE))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            t = 0
            while t < TIME_STEPS_PER_DAY:
                if not np.isnan(raw_data[t, i, j]):
                    start = t
                    while t < TIME_STEPS_PER_DAY and not np.isnan(raw_data[t, i, j]):
                        raw_events[t, i, j] = 1
                        t += 1
                    if t - start > 5:
                        raw_events[start:t, i, j] = 1
                t += 1
    return raw_events, raw_data


def event_length_legacy(raw_events: np.ndarray) -> np.ndarray:
    """
    Calculates the length of weather events.
    """
    lengths = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            events = []
            t = 0
            while t < TIME_STEPS_PER_DAY:
                if raw_events[t, i, j] == 1:
                    start = t
                    while t < TIME_STEPS_PER_DAY and raw_events[t, i, j] == 1:
                        t += 1
                    events.append((start, t - start))
                else:
                    t += 1
            lengths[i, j] = events
    return lengths
