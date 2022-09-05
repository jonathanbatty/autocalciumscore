#!/usr/bin/python3

##################################################################################################
# BSc(Hons) Computer Science                                                                     #
# CM3070 Final Project                                                                           #
# A 3-dimensional (3D), regression-based, deep learning approach to automate quantification      #
# of coronary artery calcification using thoracic computed tomography (CT) images                #
#                                                                                                #
# Submitted in partial fulfilment for BSc Computer Science | University of London | 02/09/2022   #
# Jonathan Batty | UoL Student Number: 190164315/1                                               #
##################################################################################################

# Format of command line arguments: --listOfCTscans.txt --reportFolder/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import numpy as np
import keras
import pydicom
from scipy import ndimage

# Helper functions
def resize_scan(scan_data, depth, width, height):
    """ Resize CT data across z-axis """
    
    # Set the desired size
    desired_depth = depth
    desired_width = width
    desired_height = height
        
    # Get current size
    current_depth = scan_data.shape[-1]
    current_width = scan_data.shape[0]
    current_height = scan_data.shape[1]
        
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
        
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
        
    # Resize across z-axis
    scan_data = ndimage.zoom(scan_data, (width_factor, height_factor, depth_factor), order = 1)
    return scan_data


# Check if parameters are properly formed
if len(sys.argv) == 3:

    # Get command line arguments
    input_file = sys.argv[1]
    output_filename = sys.argv[2]

    # Get directory paths from .txt file
    scan_dirs = []
    with open(input_file, 'r') as f:
        for readline in f:
            line_strip = readline.strip()
            scan_dirs.append(line_strip)
    f.close()

    # Run inference
    model = keras.models.load_model('sample_model.h5')

    with open(output_filename + '.csv', 'w') as f:
        f.write("DICOM_directory, calcium_score\n")
        for scan in scan_dirs:
            
            # Get paths of each individual DICOM file that makes up a study
            study = []
            for dirName, subdirList, fileList in os.walk(scan):
                for filename in fileList:
                    if ".dcm" in filename.lower():
                        study_id = dirName.split(os.path.sep)[0].split("/")[2]
                        study_path = dirName + "/" + filename
                        study.append(study_path)
            
            # Import dicom files as numpy matrix
            # Get first file as reference file (set pixel spacing, rows, cols, slices, etc)
            reference = pydicom.read_file(study[0])
            const_pixel_dims = (int(reference.Rows), int(reference.Columns), len(study))
            const_pixel_spacing = (float(reference.PixelSpacing[0]), float(reference.PixelSpacing[1]), float(reference.SliceThickness))
            
            # Calculate appropriate axes for the array
            x = np.arange(0.0, (const_pixel_dims[0]+1)*const_pixel_spacing[0], const_pixel_spacing[0])
            y = np.arange(0.0, (const_pixel_dims[1]+1)*const_pixel_spacing[1], const_pixel_spacing[1])
            z = np.arange(0.0, (const_pixel_dims[2]+1)*const_pixel_spacing[2], const_pixel_spacing[2])
            
            dicom_array = np.zeros(const_pixel_dims)
            
            # Loop through dicom slices
            for index, ct_slice in enumerate(study):
                
                # Import dicom file
                dicom = pydicom.read_file(ct_slice)
                
                # Check that dicom file contains pixel data:
                if "PixelData" in dicom:
                
                    # Handle possible misspecified file
                    if 'Image Storage' not in dicom.SOPClassUID.name:
                        continue

                    # Convert imported dicom image to Hounsfield unit scale
                    dicom_hu = dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept

                    # Store the slice in dicom_array
                    dicom_array[:, :, index] = dicom_hu
                            
            # Run all pre-processing steps in this section
            # Resize scan to 128 x 128 x 32 pixels using helper function above
            processed_scan = resize_scan(dicom_array, 32, 128, 128)

            # Function to run inference
            print("\n{} DICOM files were opened from: {} and resized to dimensions of {}.\n".format(len(study), scan, processed_scan.shape))

            # Mask scan with threshold of 130 HU
            processed_scan[processed_scan < 130] = 0
            processed_scan[processed_scan >= 130] = 1

            # Package in a numpy array
            scan_data = np.array([processed_scan])

            # Run inference
            print("Running coronary artery calcification assessment...")
            calcium_score = model.predict(np.expand_dims(scan_data[0], axis=0))[0][0]

            # Both write result of inference to screen and print in output .txt file
            print("\nInput DICOM path: {} had a coronary artery calcium score of      {:.2f}\n".format(scan, calcium_score))
            f.write("{}, {}\n".format(scan, calcium_score))
            

    f.close()

    print("\nThese results have been written to a .csv file at: {}.csv".format(output_filename))
    print("Have a great day!")

# Parameters are not properly formed - exit.
else:
    print("Incorrect number of arguments passed to program.")
    print("Exiting...")
    



