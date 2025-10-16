import os
import argparse
import re
from typing import Tuple, List

import pandas as pd
import numpy as np

from utils import GRID_SIZE, TIME_STEPS_PER_DAY, get_events, read_data, timer


class DatasetCreator:
    """
    Class to create a dataset for a specific year and month.
    """

    def __init__(self, year: int, month: int):
        self.year = year
        self.month = month
        self.month_names = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]

    @staticmethod
    def get_date_from_file(filename: str) -> Tuple[int, int, int]:
        match = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
        return int(match.group(1)), int(match.group(2)), int(match.group(3))

    @staticmethod
    def event_stats(raw_data: np.ndarray, i: int, j: int, start_time: int, duration: int) -> Tuple[int, float, float, float, float]:
        events = raw_data[start_time:start_time + duration, i, j]
        events = events[~np.isnan(events)]
        return (
            duration,
            np.max(events),
            np.mean(events),
            np.var(events),
            1 - np.count_nonzero(events) / duration,
        )

    def extract_df_month(self) -> pd.DataFrame:
        print(f"Listing all files for {self.year}-{self.month}.")
        files = os.listdir(f"data/raw_data/{self.year}")
        dates = [
            self.get_date_from_file(filename)
            for filename in files if filename.endswith(".npy")
        ]
        dates = [date for date in dates if date[1] == self.month]

        print(f"Total files for {self.month}/{self.year}: {len(dates)}")
        rows = []
        max_count = GRID_SIZE ** 2 * len(dates)
        count = 0
        for index, (year, month, day) in enumerate(dates):
            print(f"Processing file {
                  index + 1}/{len(dates)}: {day}/{month}/{year}")
            raw_data = read_data(year, month, day)
            events = get_events(raw_data)
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    if count % 10000 == 0:
                        print(f"{day}/{month}/{year}: {count} / {max_count}")
                    count += 1
                    for event in events[i, j]:
                        start_time, end_time = event
                        duration = end_time - start_time + 1
                        stats = self.event_stats(
                            raw_data, i, j, start_time, duration)
                        rows.append(
                            {
                                "year": year,
                                "month": month,
                                "day": day,
                                "i": i,
                                "j": j,
                                "start_time_relative": start_time,
                                "end_time_relative": end_time,
                                "start_time_absolute": start_time + TIME_STEPS_PER_DAY * (day - 1),
                                "end_time_absolute": end_time + TIME_STEPS_PER_DAY * (day - 1),
                                "duration": duration,
                                "max_intensity": stats[1],
                                "mean_intensity": stats[2],
                                "variance": stats[3],
                                "percentage_null": stats[4],
                            }
                        )
        print("Data extraction complete.")
        return pd.DataFrame(rows)

    @timer
    def write_df_month(self):
        print(f"Extracting dataframe for {self.year}, {
              self.month_names[self.month - 1]}")
        df = self.extract_df_month()
        try:
            df = df[df["max_intensity"] > 0]  # workaround
        except KeyError:
            print(f"No rain in {self.month} of {self.year}")

        directory = "data/dataframes"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = f"{directory}/{self.year}_{self.month}.pkl"
        df.to_pickle(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create dataset for a specific year and month, save to .pkl"
    )
    parser.add_argument(
        "-y", "--year", default=2021, type=int, required=False, help="The year of the dataset."
    )
    parser.add_argument(
        "-s", "--start_month", default=1, type=int, required=False, help="The start month for which the dataset will be created."
    )
    parser.add_argument(
        "-e", "--end_month", default=12, type=int, required=False, help="The end month for which the dataset will be created."
    )
    args = parser.parse_args()

    for month in range(args.start_month, args.end_month + 1):
        dc = DatasetCreator(args.year, month)
        dc.write_df_month()
