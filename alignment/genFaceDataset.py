import getMultiAlignment
import personBoxManager
import cv2
import os
import pickle
import shutil
import numpy as np


class genFaceDataset:
    def __init__(self):
        self.__getMultiAlignment = getMultiAlignment.MultifaceAlignment()
        self.__personBoxManager = personBoxManager.PersonBoxManager()
        self.__frameStep = 10
        self.personBoxDict = {}
        return

    def getFrameStep(self):
        return self.__frameStep

    def setFrameStep(self, frameStep):
        self.__frameStep = frameStep
        return "success"

    def genFromVideo(self, video, videoDir="./"):
        self.resolution = (1280, 720)
        self.videoDir = videoDir
        self.video = video
        self.faceBoxes = []
        cap = cv2.VideoCapture(self.videoDir + video)
        frameStepCounter = self.__frameStep
        frameCounter = 0
        while cap.isOpened():
            _, frame = cap.read()
            if frameStepCounter > 0:
                frameStepCounter -= 1
                continue
            frameStepCounter = self.__frameStep
            if frame is None:
                break
            frame = cv2.resize(frame, self.resolution)
            boxes = self.__getMultiAlignment.getAlignments(
                frame, mode="faceBoxes")
            for box in boxes:
                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                self.__personBoxManager.stack(box, face)
            print("\r frame: " + str(frameCounter))
            frameCounter += 1
        cap.release()
        self.personBoxDict = self.__personBoxManager.getPersonBoxDict()
        self.personImageDict = self.__personBoxManager.getPersonImageDict()

        folderPath = self.videoDir + "processed-" + self.video[:-4]
        if(os.path.exists(folderPath)):
            shutil.rmtree(folderPath)
        os.mkdir(folderPath)

        for person in self.personImageDict:
            print(person)
            subFolderPath = folderPath + "/" + str(person) + "/"
            print(subFolderPath)
            os.mkdir(subFolderPath)
            for i in range(len(self.personImageDict[person])):
                image = self.personImageDict[person][i]
                filePath = subFolderPath + str(i) + ".png"
                cv2.imwrite(filePath, image)
                # pickle.dump(image, open(filePath, "wb"))
        return "success"
