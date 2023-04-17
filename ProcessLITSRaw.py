import nibabel
import numpy as np
import pydicom
import os
from tqdm import tqdm
from PIL import Image


def convertNsave(arr,file_dir, index=0, base='',patient_num=''):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put 
    the name of each slice while using a for loop to convert all the slices
    """
    
    dicom_file = pydicom.dcmread('BaseDICOM/' + base)
    arr = arr.astype('uint16')
    dicom_file.PixelData = arr.tobytes()
    pixel_data = dicom_file.pixel_array
    pixel_data = (pixel_data/max(np.max(pixel_data), 0.000001)*255)
    if base == "litsMask.dcm":
        max_val = np.max(pixel_data)
        if max_val > 0.0:
    # Create a PIL Image from the numpy array
            image = Image.fromarray(pixel_data.astype(np.uint8))
            image.save(os.path.join(file_dir, f'p{patient_num}slice{index}.png'))
    else:
        if os.path.isfile(os.path.join("LITS/Masks", f'p{patient_num}slice{index}.png')):
            image = Image.fromarray(pixel_data.astype(np.uint8))
            image = image.rotate(-90)
            image.save(os.path.join(file_dir, f'p{patient_num}slice{index}.png'))


def nifti2dicom_1file(nifti_dir, out_dir, patient_num, base):
    """
    This function is to convert only one nifti file into dicom series

    `nifti_dir`: the path to the one nifti file
    `out_dir`: the path to output
    """

    nifti_file = nibabel.load(nifti_dir)
    nifti_array = nifti_file.get_fdata()
    number_slices = nifti_array.shape[2]

    for slice_ in tqdm(range(number_slices)):
        convertNsave(nifti_array[:,:,slice_], out_dir, slice_, patient_num=patient_num, base=base)


def nifti2dicom_mfiles(nifti_dir, out_dir='', is_mask=True, base=""):
    """
    This function is to convert multiple nifti files into dicom files

    `nifti_dir`: You enter the global path to all of the nifti files here.
    `out_dir`: Put the path to where you want to save all the dicoms here.

    PS: Each nifti file's folders will be created automatically, so you do not need to create an empty folder for each patient.
    """

    files = os.listdir(nifti_dir)
    for file in files:
        if (is_mask and "segmentation" in file) or ((not is_mask) and "volume" in file):
            spi = file.split("-")
            pn = spi[1][:-4]
            print(file, pn)
            in_path = os.path.join(nifti_dir, file)
            nifti2dicom_1file(in_path, out_dir, patient_num=pn, base=base)
            
if __name__ == "__main__":
    nifti_dir = "Rawdata\\LITS\\media\\nas\\01_Datasets\\CT\\LITS\\Training Batch 2\\"
    is_mask = False
    base = "litsImage.dcm"
    output = "LITS/Processed"
    nifti2dicom_mfiles(nifti_dir=nifti_dir, out_dir=output, is_mask=is_mask, base=base)