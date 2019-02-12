import os
import pandas as pd


def create_part_category_list(dat_directory):
    """
    Extract IDs, labels and categories from .dat files.
    Stores content to csv file

    :param dat_directory: path to .dat files
    :return: pandas dataframe with columns [id, label, category]
    """


    # read parts directory containing all .dat files
    files = []
    for file in os.listdir(dat_directory):
        if file.endswith(".dat") and not file.startswith("."):
            files.append(file)

    if len(files) == 0:
        raise ValueError("No .dat files in this directory...")

    parts = []
    for i, filename in enumerate(files):
        part_number = filename[:-4]
        with open(os.path.join(dat_directory, filename), 'r', encoding="latin-1") as f:
            first_line = f.readline()
            # print(first_line)
            if "~Moved to" in first_line:
                continue
            else:
                label = first_line[2:-1]  # skip zero and space
                if '~' in label:
                    label = label.replace('~', '')
                if label.startswith('_'):
                    label = label.replace('_', '')
                if label.startswith('='):
                    label = label.replace('=', '')
                category = label.split(' ')[0]
                parts.append([part_number, label, category])


    df = pd.DataFrame().from_records(parts, columns=["id", "label", "category"])
    df.sort_values(by=["category", "label"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":

    # example usage

    dat_directory = "../res/3d_files/complete/ldraw/parts"
    results_directory = "results"

    df = create_part_category_list(dat_directory)
    print(df["category"].value_counts())
    print(df.describe())

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    df.to_csv(os.path.join(results_directory, "parts_category_list.csv"), index=False, encoding="utf8")
