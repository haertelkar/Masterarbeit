import csv
import h5py
import numpy as np
from tqdm import tqdm

labels_csv_path = "measurements_train_7sparse_noEB_-50def/labels.csv"
hdf5_path = "measurements_train_7sparse_noEB_-50def/training_data.hdf5"
cleaned_labels_csv_path = "measurements_train_4sparse_noEB_-50def/labels_cleaned.csv"

missing_count = 0

# Open HDF5 file
with h5py.File(hdf5_path, "r") as hdf5_file:
    # Open CSV file
    with open(labels_csv_path, newline='') as csvfile, \
         open(cleaned_labels_csv_path, 'w', newline='') as cleaned_csvfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(cleaned_csvfile)
        first_row = True
        keys = []
        for row in reader:
            if first_row:
        #         # writer.writerow(row)
                first_row = False
                continue
            keys.append(row[0])
        # for cnt, row in enumerate(tqdm(reader)):
        #     
        #     keys.append(row[0])
        #     if cnt % 10000 == 0:
        #         tqdm.write(f"Processing row {cnt} of {len(keys)}")
        #         try:
        #             test = hdf5_file[keys]
        #             del test
        #         except Exception as e:
        #             # tqdm.write(f"Key {keys} not found in HDF5 file.")
        #             missing_count += 1
        #             print(e)
        #         keys = []
        if "1012698417475871071143434" not in hdf5_file.keys():
            missing_count += 1
            print(f"1012698417475871071143434 not found in HDF5 file.")
            # writer.writerow([key])
        for key in tqdm(keys):
            exit()
            if "01012698417475871071143434" not in hdf5_file.keys():
                missing_count += 1
                tqdm.write(f"Key {key} not found in HDF5 file.")
                # writer.writerow([key])
            else:
                pass
                # print(f"Key {key} found in both files.")
                # row = [key]
                # for i in range(1, len(row)):
                #     row[i] = hdf5_file[key][i-1]
            # else:
                # writer.writerow(row)

print(missing_count)