import matplotlib.pyplot as plt

def strToList(st):
    if st == '[]':
        return []
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

y_23 = []
y_24 = []
x = []
count = 1
with open("Exercises/3/videos/V67/Keypoints.txt", 'r') as k:
    for line in k:
        line = line.split("\n")[0]
        keypoints = strToList(line)
        y_23.append(keypoints[23][2]) #Extracting y-coordinates
        y_24.append(keypoints[24][2]) #Extracting y-coordinates
        x.append(count)
        count += 1
        
plt.plot(x, y_23, label = "Left hip")
plt.plot(x, y_24, label = "Right hip")

plt.xlabel("Frame number")
plt.ylabel("Y-coordinate")

plt.title("Y-coordinate Vs Frame number graph")
plt.legend()
plt.savefig("LRH_Y-Plot.png")

