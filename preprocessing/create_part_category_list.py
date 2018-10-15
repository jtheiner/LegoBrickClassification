
import os
import csv


def create_part_category_list(dat_directory):
    """
    Extract IDs, labels and categories from .dat files.
    Stores content to csv file

    :param dat_directory: path to .dat files
    :return: list of rows [id, label, category]
    """

    # read parts directory containing all .dat files
    files = []
    for file in os.listdir(dat_directory):
        if file.endswith(".dat"):
            files.append(file)

    parts = []
    for filename in files:
        part_number = filename[:-4]
        with open(os.path.join(dat_directory, filename), 'r') as f:
            first_line = f.readline()
            if "~Moved to" in first_line:
                continue
            else:
                label = first_line[2:-1] # skip zero and space
                if '~' in label:
                    label = label.replace('~', '')
                if label.startswith('_'):
                    label = label.replace('_', '')
                if label.startswith('='):
                    label = label.replace('=', '')
                category = label.split(' ')[0]
                parts.append([part_number, label, category])


    with open('results/parts_category_list.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["id", "label", "category"])
        for row in parts:
            writer.writerow(row)

    return parts




