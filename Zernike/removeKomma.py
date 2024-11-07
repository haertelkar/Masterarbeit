def remove_trailing_commas(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line.rstrip(','))

if __name__ == "__main__":
    file_path = '/data/scratch/haertelk/Masterarbeit/Zernike/measurements_train/train_labels.csv'
    remove_trailing_commas(file_path)
    file_path = '/data/scratch/haertelk/Masterarbeit/Zernike/measurements_train/vali_labels.csv'
    remove_trailing_commas(file_path)
    file_path = '/data/scratch/haertelk/Masterarbeit/Zernike/measurements_test/labels.csv'
    remove_trailing_commas(file_path)