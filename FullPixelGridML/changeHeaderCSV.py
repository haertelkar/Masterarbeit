import os
import csv


for testOrTrain in ["train", "test"]:
    folder_path = f'measurements_{testOrTrain}'  # Replace with the actual folder path

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            lines[0] = 'fileName,element1,element2,element3,element4,xAtomRel1,xAtomRel2,xAtomRel3,xAtomRel4,yAtomRel1,yAtomRel2,yAtomRel3,yAtomRel4\n'
            
            with open(file_path, 'w') as file:
                file.writelines(lines)