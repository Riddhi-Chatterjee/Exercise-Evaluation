from os.path import exists
import FeatureTemplates as FT
import cv2

def strToList(st):
    factor = -1
    for ch in st:
        if ch != '[':
            break
        factor += 1
    if factor == 0:
        return [float(x) for x in st.split("[")[1].split("]")[0].split(", ")]
    
    sList = [x+("]"*factor) if x[len(x) - 1] != ']' else x for x in st[1:len(st)-1].split("]"*factor + ", ")]
    lst = []
    for s in sList:
        lst.append(strToList(s))
    return lst

visThreshold = 0.5
object_dispatcher = {
    '2d': FT.distance_2D,
    '2k': FT.keypoint_2D,
    '2a': FT.angle_2D,
    '2v': FT.velocity_2D
    #'3d': FT.distance_3D,
    #'3k': FT.keypoint_3D,
    #'3a': FT.angle_3D,
    #'3v': FT.velocity_3D
}

exercise = int(input("Enter exercise number: "))
features = []
if exists("Exercises/"+str(exercise)+"/EssentialFeatures.json"):
    with open("Exercises/"+str(exercise)+"/EssentialFeatures.json", "r") as ef:
        for line in ef:
            line = line.split("\n")[0].lower()
            components = line.split(", ")
            featureType = components[0] + components[1]
            parameters = [int(x) if x.isdigit() else x for x in components[2:]]
            features.append(object_dispatcher[featureType](parameters, True, visThreshold))

if exists("Exercises/"+str(exercise)+"/NonEssentialFeatures.json"):
    with open("Exercises/"+str(exercise)+"/NonEssentialFeatures.json", "r") as nef:
        for line in nef:
            line = line.split("\n")[0].lower()
            components = line.split(", ")
            featureType = components[0] + components[1]
            parameters = [int(x) if x.isdigit() else x for x in components[2:]]
            features.append(object_dispatcher[featureType](parameters, False, visThreshold))
        
video = 1
while(exists("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/V"+str(video)+".mp4") and exists("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Keypoints.txt")):
    with open("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Features.txt", "w") as f:
        with open("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Keypoints.txt", "r") as k:
            o_fps = 5 #Dummy initialisation
            cap = cv2.VideoCapture("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/V"+str(video)+".mp4")
        
            # Finding OpenCV version:
            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

            if int(major_ver)  < 3 :
                o_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS) 
            else :
                o_fps = cap.get(cv2.CAP_PROP_FPS)
            
            for frameKeypoints in k:
                frameKeypoints = strToList(frameKeypoints.split("\n")[0])
                validFrame = True
                frameFeatures = []
                for feature in features:
                    feature.loadData(frameKeypoints)
                    feature.calculate(video, o_fps)
                    if feature.isEssential == True and validFrame == True:
                        for v in feature.value:
                            if v == "None":
                                validFrame = False
                                break
                    #if not validFrame:
                    #    break
                
                    s = ""
                    for p in feature.original_parameters:
                        s += ", "
                        s += str(p)
                        
                    descriptor = feature.type[0]+", "+feature.type[1]+s
                    frameFeatures.append([descriptor, feature.value])
                if validFrame:
                    f.write(str(frameFeatures)+"\n")
    video += 1
    