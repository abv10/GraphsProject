import os
import random

def split_into_folds(path, name):
    # Get a list of all the files in the directory
    files = os.listdir(path)
    # Create a dictionary to group files by patient number
    patients = {}
    for file in files:
        if file.startswith('p') and file.endswith('.png'):
            patient = file[1:file.index("slice")+1]
            if patient not in patients:
                patients[patient] = []
            patients[patient].append(file)
    # Shuffle the patient groups
    groups = list(patients.values())
    random.shuffle(groups)
    # Split the groups into folds
    folds = [[] for _ in range(6)]
    for group in groups:
        # Determine the smallest fold
        smallest_fold = min(folds, key=len)
        # Add the group to the smallest fold
        smallest_fold.extend(group)
    # Write the folds to files
    for i, fold in enumerate(folds):
        if i < 5:
            with open(f"{name}_fold_{i}.txt", "w") as f:
                f.write("\n".join(fold))
        else:
            with open(f"{name}_test.txt", "w") as f:
                f.write("\n".join(fold))





def create_train_validation_split(name, val_fold):
    # Read the file names from the folds
    folds = []
    for i in range(5):
        with open(f"{name}_fold_{i}.txt", "r") as f:
            fold = f.read().splitlines()
            folds.append(fold)
    # Get the file names for the validation fold
    val_files = folds[val_fold]
    # Get the file names for the training folds
    train_files = []
    for i in range(5):
        if i != val_fold:
            train_files.extend(folds[i])
    # Return the file name arrays
    return train_files, val_files



if __name__ == "__main__":
    split_into_folds("LITS/Masks", "lits")
