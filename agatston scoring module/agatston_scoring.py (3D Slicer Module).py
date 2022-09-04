# The below function can be executed  in 3D slicer following segmentation of coronary calcium to obtain the Agatston score.
# Adapted from: https://gist.github.com/lassoan/d85be45b7284a3b4e580688fccdb1d02

def computeAgatstonScore(volumeNode, minimumIntensityThreshold = 130, minimumIslandSizeInMm2 = 1.0, verbose = False):
    """ Returns the Agatston score based on segmentation of coronary calcification """
    import numpy as np
    import math
    import SimpleITK as sitk

    # Get required properties from DICOM file and input parameters
    voxelArray = slicer.util.arrayFromVolume(volumeNode)
    areaOfPixelMm2 = volumeNode.GetSpacing()[0] * volumeNode.GetSpacing()[1]
    minimumIslandSizeInPixels = int(round(minimumIslandSizeInMm2/areaOfPixelMm2))
    numberOfSlices = voxelArray.shape[0]

    # Initialise total Agatston score
    totalScore = 0

    # Iterate over each slice in scan; calculate score for each slice
    for sliceIndex in range(numberOfSlices):
        voxelArraySlice = voxelArray[sliceIndex]
        maxIntensity = voxelArraySlice.max()

        if maxIntensity < minimumIntensityThreshold:
            continue

        # Get a thresholded image to analyse islands (connected components)
        # If island size less than minimum size then it will be discarded.
        thresholdedVoxelArraySlice = voxelArraySlice>minimumIntensityThreshold
        sliceImage = sitk.GetImageFromArray(voxelArraySlice)
        thresholdedSliceImage = sitk.GetImageFromArray(thresholdedVoxelArraySlice.astype(int))
        labelImage = sitk.ConnectedComponentImageFilter().Execute(thresholdedSliceImage)
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(sliceImage, labelImage)
        numberOfNonZeroVoxels = 0
        numberOfIslands = 0
        sliceScore = 0

        for labelIndex in stats.GetLabels():
            if labelIndex == 0:
                continue
            if stats.GetCount(labelIndex) < minimumIslandSizeInPixels:
                continue

        maxIntensity = stats.GetMaximum(labelIndex)
        weightFactor = math.floor(maxIntensity/100)

        if weightFactor > 4:
            weightFactor = 4.0

        numberOfNonZeroVoxelsInIsland = stats.GetCount(labelIndex)
        sliceScore += numberOfNonZeroVoxelsInIsland * areaOfPixelMm2 * weightFactor
        numberOfNonZeroVoxels += numberOfNonZeroVoxelsInIsland
        numberOfIslands += 1
        totalScore += sliceScore

        if (sliceScore > 0) and verbose:
            print("Slice {0} score: {1:.1f} (from {2} islands of size > {3}, {4} voxels)".format(
                sliceIndex, sliceScore, numberOfIslands, minimumIslandSizeInPixels, numberOfNonZeroVoxels))

        slicer.app.processEvents()

    return totalScore

print("Agatston score: {0}".format(computeAgatstonScore(getNode('masked'), verbose = True)))