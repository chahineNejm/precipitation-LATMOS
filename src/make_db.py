import pandas as pd
import sqlite3
import os

if __name__ == "__main__":
    folder_path = "data/dataframes"
    database_path = "precip.db"

    pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]

    with sqlite3.connect(database_path) as conn:
        conn.execute("DROP TABLE IF EXISTS events")
        for i, pkl_file in enumerate(pkl_files):
            file_path = os.path.join(folder_path, pkl_file)
            df = pd.read_pickle(file_path)

            df.to_sql(name="events", con=conn, if_exists="append", index=False)
            print(f"Inserted {i+1} of {len(pkl_files)}: {pkl_file}")

    print("All .pkl files have been successfully inserted into the SQLite database.")
