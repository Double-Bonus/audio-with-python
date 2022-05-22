import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

CHECK_BOX_W = 16
CAT_LABELS_W = 20

#callbacks
def Item_test():
    for i in range(0,3):
        if animalItems[i].get():
            print("Animal:", i, "is selected")
        else:
            print("Animal:", i, "is not selected")
            
        if natureItems[i].get():
            print("Nature:", i, "is selected")
        else:
            print("Nature:", i, "is not selected")          
            
    print("\n")

soundCategories = ["Gyvunų garsai", "Aplinkos ir vandens garsai", "Žmonių, nekalbiniai garsai", "Buitiniai garsai", "Miesto garsai"]

animalsStr = [
"Šuo",
"Gaidys",
"Kiaulė",
"Karvė",
"Varlė",
"Katė",
"Višta",
"Vabzdžiai",
"Avys",
"Varna"]

natureWaterStr = [
"Lietus",
"Jūros bangos",
"Ugnies spragsėjimas",
"Svirpliai",
"Paukščių čiulbėjimas",
"Vandens lašai",
"Vėjas",
"Vandens pylimas",
"Tualeto nuleidimas",
"Perkūnija"
]

humanNon_speechStr = [
"Kūdikio verkimas",
"Čiaudėjimas",
"Plojimai",
"Kvėpavimas",
"Kosėjimas",
"Žingsniai",
"Juokas",
"Dantų valymas",
"Knarkimas",
"Gerimas"
]

buitiniaiStr = [
"Durų beldimas",
"Komp. pelės paspaudimas",
"Rašymas klaviatūra",
"Durų girgždėjimas",
"Skardinės atidarymas",
"Skalbimo mašina",
"Dulkių siurblys",
"Laikrodžio žadintuvas",
"Laikrodžio tiksėjimas",
"Stiklo dūžimas"    
]

miestoGarsaiStr = [
"Sraigtasparnis",
"Grandininis pjūklas",
"Sirena",
"Automobilio signalas",
"Variklis",
"Traukinys",
"Bažnyčios varpai",
"Lėktuvas",
"Fejerverkai",
"Rankinis pjūklas",
]


top = tk.Tk()
top.title('Audio demo')
top.geometry('1400x800')


labelDetection = tk.Label(top, text = "Pasirinktų garsų detektavimas")
labelDetection.config(font =("Ariel", 20))
labelDetection.grid(row=0, column=2)


startingRow = 1
###############################################################################
label_1 = tk.Label(top, text = soundCategories[0], font =("Ariel", 12),width=CAT_LABELS_W, anchor="center")
label_1.grid(row=0+startingRow, column=0)

animalItems = []
# appending instances to list 
for i in range(10):
    animalItems.append(tk.IntVar())
    
for i in range(10):
    cb1 = tk.Checkbutton(top, text=animalsStr[i], variable=animalItems[i], command=Item_test, anchor="w", width=CHECK_BOX_W)
    cb1.grid(row=i+1+startingRow, column=0)

###############################################################################
label_2 = tk.Label(top, text = soundCategories[1], font =("Ariel", 12),width=CAT_LABELS_W, anchor="center")
label_2.grid(row=0+startingRow, column=1)

natureItems = [] 
for i in range(10):
    natureItems.append(tk.IntVar())
for i in range(10):
    cb1 = tk.Checkbutton(top, text=natureWaterStr[i], variable=natureItems[i], width=CHECK_BOX_W, anchor="w", command=Item_test)
    cb1.grid(row=i+1+startingRow, column=1)


###############################################################################
label_3 = tk.Label(top, text = soundCategories[2], font =("Ariel", 12),width=CAT_LABELS_W, anchor="center")
label_3.grid(row=0+startingRow, column=2)

humanItems = [] 
for i in range(10):
    humanItems.append(tk.IntVar())
for i in range(10):
    cb1 = tk.Checkbutton(top, text=humanNon_speechStr[i], variable=humanItems[i], command=Item_test, anchor="w", width=CHECK_BOX_W)
    cb1.grid(row=i+1+startingRow, column=2)
    
    
    
###############################################################################
label_4 = tk.Label(top, text = soundCategories[3], font =("Ariel", 12),width=CAT_LABELS_W, anchor="center")
label_4.grid(row=0+startingRow, column=3)

domesticItems = [] 
for i in range(10):
    domesticItems.append(tk.IntVar())
for i in range(10):
    cb1 = tk.Checkbutton(top, text=buitiniaiStr[i], variable=domesticItems[i], command=Item_test, anchor="w", width=CHECK_BOX_W)
    cb1.grid(row=i+1+startingRow, column=3)
    
###############################################################################
label_5 = tk.Label(top, text = soundCategories[4], font =("Ariel", 12),width=CAT_LABELS_W, anchor="center")
label_5.grid(row=0+startingRow, column=4)

urbanItems = [] 
for i in range(10):
    urbanItems.append(tk.IntVar())
for i in range(10):
    cb1 = tk.Checkbutton(top, text=miestoGarsaiStr[i], variable=urbanItems[i], command=Item_test, anchor="w", width=CHECK_BOX_W)
    cb1.grid(row=i+1+startingRow, column=4)
###############################################################################
endRow = startingRow + 11
    
################### Detection  ################################################
probLabel = tk.Label(top, text="Mažiausia tikimybė (procentais):")
probValue = tk.StringVar(top, value='70')
probEntry = tk.Entry(top, textvariable=probValue)


probLabel.grid(row=endRow, column=0)
probEntry.grid(row=endRow, column=1)

button1 = tk.Button(top, text="Detektuoti", height = 6, width = 20, command=Item_test)
button1.grid(row=endRow+1, column=0)

detectStateLabel = tk.Label(top, text="Detektavimo būsena:", font =("Ariel", 12))
inputStateLabel = tk.Label(top, text="nepradėta", bg="yellow", font =("Ariel", 12)) # neaptiktas aptiktas
# inputStateLabel.config(bg="yellow")
detectStateLabel.grid(row=endRow+1, column=2)
inputStateLabel.grid(row=endRow+1, column=3)

################### Classitfication  ################################################
clasfRow = endRow+2

labelDetection = tk.Label(top, text = "Dabartinio garso klasifikavimas")
labelDetection.config(font =("Ariel", 20))
labelDetection.grid(row=clasfRow, column=2)

clasfRow += 1
btnClasf = tk.Button(top, text="Klasifikuoti", height = 6, width = 20, command=Item_test)
btnClasf.grid(row=clasfRow, column=0)

soundClss_lb = tk.Label(top, text="Garso klasė:", font =("Ariel", 12))
soundClss_lb.grid(row=clasfRow, column=1)

currentSoundClass_lb = tk.Label(top, text="------", font =("Ariel", 12))
currentSoundClass_lb.grid(row=clasfRow, column=2)

clasfProb_lb = tk.Label(top, text="Tikimybė", font =("Ariel", 12))
clasfProb_lb.grid(row=clasfRow, column=3)

currentClasfProb_lb = tk.Label(top, text="0%", font =("Ariel", 12))
currentClasfProb_lb.grid(row=clasfRow, column=4)


top.mainloop()
