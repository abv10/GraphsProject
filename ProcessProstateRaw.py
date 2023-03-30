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
    
# Create a PIL Image from the numpy array
    image = Image.fromarray(pixel_data)
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


def nifti2dicom_mfiles(nifti_dir, out_dir=''):
    """
    This function is to convert multiple nifti files into dicom files

    `nifti_dir`: You enter the global path to all of the nifti files here.
    `out_dir`: Put the path to where you want to save all the dicoms here.

    PS: Each nifti file's folders will be created automatically, so you do not need to create an empty folder for each patient.
    """

    files = os.listdir(nifti_dir)
    for file in files:
        in_path = os.path.join(nifti_dir, file)
        out_path = os.path.join(out_dir, file)
        os.mkdir(out_path)
        nifti2dicom_1file(in_path, out_path)
            
