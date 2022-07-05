import openpyxl
from pathlib import Path
from os.path import exists
import matplotlib.pyplot as plt

def readExerciseTS(path, exercise):
    cell = {
        1 : "B2",
        2 : "C2",
        3 : "D2",
        4 : "E2",
        5 : "F2"
    }
    xlsx_file = Path(path)
    wb_obj = openpyxl.load_workbook(xlsx_file) 

    # Read the active sheet:
    sheet = wb_obj.active
    return sheet[cell[exercise]].value

exercise = int(input("Enter exercise number: "))

people = []
scores = []
labels = ["E_ID", "NE_ID", "B_ID", "P_ID", "S_ID"]
for label in labels:
    i=1
    while(exists("Scores/"+label+str(i)+".xlsx")):
        TS = readExerciseTS("Scores/"+label+str(i)+".xlsx", exercise)
        if TS != None:
            people.append(label+str(i))
            scores.append(TS)
        i += 1

fig = plt.figure(figsize = (70, 35))
 
# creating the bar plot
plt.bar(people, scores, color ='maroon')
 
plt.xlabel("People")
plt.ylabel("Total Scores")
plt.title("Total scores for Exercise-"+str(exercise))
plt.savefig("TS_Plot_Ex"+str(exercise)+".png")