from mtCNN.faceDetectorAndAlignment import faceDetectorAndAlignment
import cv2
import numpy as np
import torch


# def getMultiAlignment(image, isImage=False):
#     if isImage:
#         inputImage = image
#     else:
#         inputImage = cv2.imread(image)
#     faceBoxes, alignedFaces = detectorAndAlignment.run(inputImage)
#     if alignedFaces.shape[0] != 0:
#         return alignedFaces
#     return

class MultifaceAlignment:
    def __init__(self):
        self.detectorAndAlignment = faceDetectorAndAlignment(pNetWightFile='pNetModel.pth',
                                                             rNetWightFile='rNetModel.pth', oNetWightFile='oNetModel.pth')

    def getAlignments(self, image, mode="alignedFaces"):
        # image = cv2.imread(image)
        if image is None:
            return "fail"
        faceBoxes, alignedFaces = self.detectorAndAlignment.run(image)
        if mode == "all":
            return faceBoxes, alignedFaces
        elif mode == "alignedFaces":
            return alignedFaces
        else:
            return faceBoxes
