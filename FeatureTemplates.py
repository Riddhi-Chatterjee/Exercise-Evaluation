import math
import numpy as np
import time

class feature:
    
    def __init__(self, parameters, isEssential, visThreshold):
        self.parameters = parameters
        self.isEssential = isEssential
        self.visThreshold = visThreshold
        self.keypoints = []
        self.value = ["None"]
        
    def checkVisibility(self):
        allVisible = True
        for parameter in self.parameters:
            if self.keypoints[parameter][4] < self.visThreshold:
                allVisible = False
                break
        if not allVisible:
            self.value = ["None" for x in self.value]
        return allVisible
        

class distance_2D(feature):

    def __init__(self, parameters, isEssential, visThreshold):
        feature.__init__(self, parameters, isEssential, visThreshold)
        self.type = '2d'

    def loadData(self, keypoints):
        self.keypoints = keypoints
    
    def calculate(self, video):
        if len(self.parameters) != 2 and len(self.parameters) != 4:
            return
        
        if len(self.parameters) == 2:

            first_point = []
            first_point.append(self.keypoints[self.parameters[0]][1])
            first_point.append(self.keypoints[self.parameters[0]][2])

            second_point = []
            second_point.append(self.keypoints[self.parameters[1]][1])
            second_point.append(self.keypoints[self.parameters[1]][2])
        
            distance = math.sqrt((first_point[0]-second_point[0])**2 + (first_point[1]-second_point[1])**2)

            self.value = [distance]
        
        elif len(self.parameters) == 4:
            
            first_point = []
            first_point.append(self.keypoints[self.parameters[0]][1])
            first_point.append(self.keypoints[self.parameters[0]][2])

            second_point = []
            second_point.append(self.keypoints[self.parameters[1]][1])
            second_point.append(self.keypoints[self.parameters[1]][2])
        
            distance1 = math.sqrt((first_point[0]-second_point[0])**2 + (first_point[1]-second_point[1])**2)

            third_point = []
            third_point.append(self.keypoints[self.parameters[2]][1])
            third_point.append(self.keypoints[self.parameters[2]][2])

            fourth_point = []
            fourth_point.append(self.keypoints[self.parameters[3]][1])
            fourth_point.append(self.keypoints[self.parameters[3]][2])
        
            distance2 = math.sqrt((third_point[0]-fourth_point[0])**2 + (third_point[1]-fourth_point[1])**2)

            ratio = "None"
            if distance2 != 0:
                ratio = distance1/distance2
            
            self.value = [ratio]
            
        self.checkVisibility()


class keypoint_2D(feature):

    def __init__(self, parameters, isEssential, visThreshold):
        feature.__init__(self, parameters, isEssential, visThreshold)
        self.type = '2k'

    def loadData(self, keypoints):
        self.keypoints = keypoints

    def calculate(self, video):
        if len(self.parameters) != 1:
            return
        
        keypoint = self.keypoints[self.parameters[0]]
        self.value = keypoint[1:3] + keypoint[4:5]
        self.checkVisibility()


class angle_2D(feature):

    def __init__(self, parameters, isEssential, visThreshold):
        feature.__init__(self, parameters, isEssential, visThreshold)
        self.type = '2a'

    def loadData(self, keypoints):
        self.keypoints = keypoints

    def calculate(self, video):
        if len(self.parameters) != 3:
            return

        first_point = []
        first_point.append(self.keypoints[self.parameters[0]][1])
        first_point.append(self.keypoints[self.parameters[0]][2])

        second_point = []
        second_point.append(self.keypoints[self.parameters[1]][1])
        second_point.append(self.keypoints[self.parameters[1]][2])

        third_point = []
        
        if type(self.parameters[2]) == int:
            third_point.append(self.keypoints[self.parameters[2]][1])
            third_point.append(self.keypoints[self.parameters[2]][2])

        elif self.parameters[2].lower() == 'x':
            third_point.append(self.keypoints[self.parameters[1]][1] + 1)
            third_point.append(self.keypoints[self.parameters[1]][2])

        elif self.parameters[2].lower() == 'y':
            third_point.append(self.keypoints[self.parameters[1]][1])
            third_point.append(self.keypoints[self.parameters[1]][2] + 1)
        
        A = np.array([(first_point[0]-second_point[0]),(first_point[1]-second_point[1])])
        B = np.array([(third_point[0]-second_point[0]),(third_point[1]-second_point[1])])
        
        modA = np.linalg.norm(A)
        modB = np.linalg.norm(B)
        
        dotProd = np.dot(A, B)
        crossProd = np.cross(A,B)
        modCrossProd = np.linalg.norm(crossProd)
        
        ang = ["None"]
        dir = ["None"]
        if modA == 0 or modB == 0:
            ang = ["None"]
            dir = ["None"]
        else:
            if modCrossProd != 0:
                dir = list(crossProd/modCrossProd)
            else:
                dir = ["None"]
               
            cos = dotProd/(modA*modB) 
            if cos >= -1 and cos <= 1:
                ang = [self.toDegree(math.acos(cos))]
            else:
                ang = ["None"]
                dir = ["None"]
                
        if self.parameters[3].lower() == 'd':
            self.value = ang + dir
        elif self.parameters[3].lower() == 'nd':
            self.value = ang
            
        self.checkVisibility()
    
    def toDegree(self, ang):
        return ang*(180/math.pi)   
    
class velocity_2D(feature):
    
    def __init__(self, parameters, isEssential, visThreshold):
        feature.__init__(self, parameters, isEssential, visThreshold)
        self.type = '2v'
        self.video = -1
        self.prevTime = -1
        self.currTime = -1
        self.prevKeypoints = []
        self.currKeypoints = []
        
    def loadData(self, keypoints):
        self.keypoints = keypoints
        
    def calculate(self, video): #Currently calculating velocities in pixels/ms
        if len(self.parameters) != 1 and len(self.parameters) != 2 and len(self.parameters) != 3 and len(self.parameters) != 4:
            return
        
        if self.checkVisibility():
            if video != self.video:
                self.prevTime = time.time()*1000000
                self.prevKeypoints = self.keypoints
                self.video = video
                self.value = ["None", "None"]

            else:
                self.currTime = time.time()*1000000
                self.currKeypoints = self.keypoints

                tDiffMili = (self.currTime - self.prevTime)/1000
                
                if len(self.parameters) == 1:
                    prevPoint = []
                    prevPoint.append(self.prevKeypoints[self.parameters[0]][1])
                    prevPoint.append(self.prevKeypoints[self.parameters[0]][2])
                    prevPoint = np.array(prevPoint)
                    
                    currPoint = []
                    currPoint.append(self.currKeypoints[self.parameters[0]][1])
                    currPoint.append(self.currKeypoints[self.parameters[0]][2])
                    currPoint = np.array(currPoint)
                    
                    d = currPoint - prevPoint
                    self.value = list(d/tDiffMili)
                    
                elif len(self.parameters) == 2:
                    pass
                
                elif len(self.parameters) == 3:
                    pass
                
                elif len(self.parameters) == 4: #Scaled velocity; Sample params: [0, 'r', 1, 2]
                    pass
                
                self.prevTime = self.currTime
                self.prevKeypoints = self.currKeypoints
        else:
            self.value = ["None", "None"]