from tkinter import ttk

import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import tkinter as tk
from screeninfo import get_monitors
import numpy as np
from tabulate import tabulate
from joblib import dump, load

df = pd.read_csv("wdbc.data",names=["diagnosis","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"])
model = LogisticRegression(max_iter=10000)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


root = tk.Tk()
root.title("Projekt")




screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height
root.geometry(f"{int(screen_width/2)}x{int(screen_height/2)}")

left_frame = tk.Frame(root,borderwidth=4, relief="ridge",width=int(screen_width / 8),height=int(screen_width / 4))
left_frame.pack(side="left",padx=10, pady=10)
left_frame.pack_propagate(0)
right_frame = tk.Frame(root,width=int(screen_width / 4), height=left_frame["height"],borderwidth=4, relief="ridge")
right_frame.pack(side="right",padx=10, pady=10)
right_frame.pack_propagate(0)

label = tk.Label(right_frame, text="")
label.pack(padx=10, pady=10)

def buildModel():
    global X,y,model
    X = preprocessing.normalize(X)
    model.fit(X, y)

buttonBuildModel = tk.Button(left_frame, text="Train", command=buildModel)
buttonBuildModel.pack()

def testModel():
    global X,y,model
    tempList = cross_val_score(estimator=model, X=X, y=y, cv=LeaveOneOut(), scoring="accuracy")
    wynik8 = tempList.mean()
    print(wynik8)

buttonTestModel = tk.Button(left_frame, text="Test", command=testModel)
buttonTestModel.pack()
def open_new_window():
    new_window = tk.Toplevel(root)
    new_window.title("Test new data")
    l1 = ttk.Label(new_window, text="1")
    l1.pack()
    k1 = ttk.Entry(new_window)
    k1.pack()
    l2 = ttk.Label(new_window, text="2")
    l2.pack()
    k2 = ttk.Entry(new_window)
    k2.pack()
    l3 = ttk.Label(new_window, text="3")
    l3.pack()
    k3 = ttk.Entry(new_window)
    k3.pack()
    l4 = ttk.Label(new_window, text="4")
    l4.pack()
    k4 = ttk.Entry(new_window)
    k4.pack()
    l5 = ttk.Label(new_window, text="5")
    l5.pack()
    k5 = ttk.Entry(new_window)
    k5.pack()
    l6 = ttk.Label(new_window, text="6")
    l6.pack()
    k6 = ttk.Entry(new_window)
    k6.pack()
    l7 = ttk.Label(new_window, text="7")
    l7.pack()
    k7 = ttk.Entry(new_window)
    k7.pack()
    l8 = ttk.Label(new_window, text="8")
    l8.pack()
    k8 = ttk.Entry(new_window)
    k8.pack()
    l9 = ttk.Label(new_window, text="9")
    l9.pack()
    k9 = ttk.Entry(new_window)
    k9.pack()
    l10 = ttk.Label(new_window, text="10")
    l10.pack()
    k10= ttk.Entry(new_window)
    k10.pack()
    l11 = ttk.Label(new_window, text="11")
    l11.pack()
    k11 = ttk.Entry(new_window)
    k11.pack()
    l12 = ttk.Label(new_window, text="12")
    l12.pack()
    k12 = ttk.Entry(new_window)
    k12.pack()
    l13 = ttk.Label(new_window, text="13")
    l13.pack()
    k13 = ttk.Entry(new_window)
    k13.pack()
    l14 = ttk.Label(new_window, text="14")
    l14.pack()
    k14 = ttk.Entry(new_window)
    k14.pack()
    l15 = ttk.Label(new_window, text="15")
    l15.pack()
    k15 = ttk.Entry(new_window)
    k15.pack()
    l16 = ttk.Label(new_window, text="16")
    l16.pack()
    k16 = ttk.Entry(new_window)
    k16.pack()
    l17 = ttk.Label(new_window, text="17")
    l17.pack()
    k17 = ttk.Entry(new_window)
    k17.pack()
    l18 = ttk.Label(new_window, text="18")
    l18.pack()
    k18= ttk.Entry(new_window)
    k18.pack()
    l19 = ttk.Label(new_window, text="19")
    l19.pack()
    k19 = ttk.Entry(new_window)
    k19.pack()
    l20 = ttk.Label(new_window, text="20")
    l20.pack()
    k20 = ttk.Entry(new_window)
    k20.pack()
    l21 = ttk.Label(new_window, text="21")
    l21.pack()
    k21 = ttk.Entry(new_window)
    k21.pack()
    l22 = ttk.Label(new_window, text="22")
    l22.pack()
    k22 = ttk.Entry(new_window)
    k22.pack()
    l23 = ttk.Label(new_window, text="23")
    l23.pack()
    k23 = ttk.Entry(new_window)
    k23.pack()
    l24 = ttk.Label(new_window, text="24")
    l24.pack()
    k24 = ttk.Entry(new_window)
    k24.pack()
    l25 = ttk.Label(new_window, text="25")
    l25.pack()
    k25 = ttk.Entry(new_window)
    k25.pack()
    l26= ttk.Label(new_window, text="26")
    l26.pack()
    k26 = ttk.Entry(new_window)
    k26.pack()
    l27 = ttk.Label(new_window, text="27")
    l27.pack()
    k27 = ttk.Entry(new_window)
    k27.pack()
    l28 = ttk.Label(new_window, text="28")
    l28.pack()
    k28 = ttk.Entry(new_window)
    k28.pack()
    l29 = ttk.Label(new_window, text="29")
    l29.pack()
    k29 = ttk.Entry(new_window)
    k29.pack()
    l30 = ttk.Label(new_window, text="30")
    l30.pack()
    k30 = ttk.Entry(new_window)
    k30.pack()
    def add_new():
        global model
        list =np.array([float(k1.get()),float(k2.get()),float(k3.get()),float(k4.get()),float(k5.get()),float(k6.get()),float(k7.get()),float(k8.get()),float(k9.get()),float(k10.get()),float(k11.get()),float(k12.get()),float(k13.get()),float(k14.get()),float(k15.get()),float(k16.get()),float(k17.get()),float(k18.get()),float(k19.get()),float(k20.get()),float(k21.get()),float(k22.get()),float(k23.get()),float(k24.get()),float(k25.get()),float(k26.get()),float(k27.get()),float(k28.get()),float(k29.get()),float(k30.get())])
        list = list.reshape(1,-1)

        ans= model.predict(list)

        print(ans)
    add_button = ttk.Button(new_window, text="Dodaj", command=add_new)
    add_button.pack()



buttonPredict = tk.Button(left_frame, text="test data", command=open_new_window)
buttonPredict.pack()




def open_new_window_add_new():
    new_window = tk.Toplevel(root)
    new_window.title("add new data")



    lans = ttk.Label(new_window, text="answer")
    lans.pack()
    kans = ttk.Entry(new_window)
    kans.pack()

    l1 = ttk.Label(new_window, text="1")
    l1.pack()
    k1 = ttk.Entry(new_window)
    k1.pack()
    l2 = ttk.Label(new_window, text="2")
    l2.pack()
    k2 = ttk.Entry(new_window)
    k2.pack()
    l3 = ttk.Label(new_window, text="3")
    l3.pack()
    k3 = ttk.Entry(new_window)
    k3.pack()
    l4 = ttk.Label(new_window, text="4")
    l4.pack()
    k4 = ttk.Entry(new_window)
    k4.pack()
    l5 = ttk.Label(new_window, text="5")
    l5.pack()
    k5 = ttk.Entry(new_window)
    k5.pack()
    l6 = ttk.Label(new_window, text="6")
    l6.pack()
    k6 = ttk.Entry(new_window)
    k6.pack()
    l7 = ttk.Label(new_window, text="7")
    l7.pack()
    k7 = ttk.Entry(new_window)
    k7.pack()
    l8 = ttk.Label(new_window, text="8")
    l8.pack()
    k8 = ttk.Entry(new_window)
    k8.pack()
    l9 = ttk.Label(new_window, text="9")
    l9.pack()
    k9 = ttk.Entry(new_window)
    k9.pack()
    l10 = ttk.Label(new_window, text="10")
    l10.pack()
    k10= ttk.Entry(new_window)
    k10.pack()
    l11 = ttk.Label(new_window, text="11")
    l11.pack()
    k11 = ttk.Entry(new_window)
    k11.pack()
    l12 = ttk.Label(new_window, text="12")
    l12.pack()
    k12 = ttk.Entry(new_window)
    k12.pack()
    l13 = ttk.Label(new_window, text="13")
    l13.pack()
    k13 = ttk.Entry(new_window)
    k13.pack()
    l14 = ttk.Label(new_window, text="14")
    l14.pack()
    k14 = ttk.Entry(new_window)
    k14.pack()
    l15 = ttk.Label(new_window, text="15")
    l15.pack()
    k15 = ttk.Entry(new_window)
    k15.pack()
    l16 = ttk.Label(new_window, text="16")
    l16.pack()
    k16 = ttk.Entry(new_window)
    k16.pack()
    l17 = ttk.Label(new_window, text="17")
    l17.pack()
    k17 = ttk.Entry(new_window)
    k17.pack()
    l18 = ttk.Label(new_window, text="18")
    l18.pack()
    k18= ttk.Entry(new_window)
    k18.pack()
    l19 = ttk.Label(new_window, text="19")
    l19.pack()
    k19 = ttk.Entry(new_window)
    k19.pack()
    l20 = ttk.Label(new_window, text="20")
    l20.pack()
    k20 = ttk.Entry(new_window)
    k20.pack()
    l21 = ttk.Label(new_window, text="21")
    l21.pack()
    k21 = ttk.Entry(new_window)
    k21.pack()
    l22 = ttk.Label(new_window, text="22")
    l22.pack()
    k22 = ttk.Entry(new_window)
    k22.pack()
    l23 = ttk.Label(new_window, text="23")
    l23.pack()
    k23 = ttk.Entry(new_window)
    k23.pack()
    l24 = ttk.Label(new_window, text="24")
    l24.pack()
    k24 = ttk.Entry(new_window)
    k24.pack()
    l25 = ttk.Label(new_window, text="25")
    l25.pack()
    k25 = ttk.Entry(new_window)
    k25.pack()
    l26= ttk.Label(new_window, text="26")
    l26.pack()
    k26 = ttk.Entry(new_window)
    k26.pack()
    l27 = ttk.Label(new_window, text="27")
    l27.pack()
    k27 = ttk.Entry(new_window)
    k27.pack()
    l28 = ttk.Label(new_window, text="28")
    l28.pack()
    k28 = ttk.Entry(new_window)
    k28.pack()
    l29 = ttk.Label(new_window, text="29")
    l29.pack()
    k29 = ttk.Entry(new_window)
    k29.pack()
    l30 = ttk.Label(new_window, text="30")
    l30.pack()
    k30 = ttk.Entry(new_window)
    k30.pack()
    def add_new():

        df.loc[len(df.index)] =[kans.get(),float(k1.get()),float(k2.get()),float(k3.get()),float(k4.get()),float(k5.get()),float(k6.get()),float(k7.get()),float(k8.get()),float(k9.get()),float(k10.get()),float(k11.get()),float(k12.get()),float(k13.get()),float(k14.get()),float(k15.get()),float(k16.get()),float(k17.get()),float(k18.get()),float(k19.get()),float(k20.get()),float(k21.get()),float(k22.get()),float(k23.get()),float(k24.get()),float(k25.get()),float(k26.get()),float(k27.get()),float(k28.get()),float(k29.get()),float(k30.get())]




    add_button = ttk.Button(new_window, text="Add data", command=add_new)
    add_button.pack()



buttonAddNew = tk.Button(left_frame, text="add data", command=open_new_window_add_new)
buttonAddNew.pack()


def showTable():
    print(tabulate(df, headers='keys', tablefmt='psql'))


buttonShowTable = tk.Button(left_frame, text="Show table", command=showTable)
buttonShowTable.pack()

def save():
    dump(model, 'model_save.joblib')


buttonSave = tk.Button(left_frame, text="Save model", command=save)
buttonSave.pack()

def importModel():
    model = load('model_save.joblib')


buttonImport = tk.Button(left_frame, text="Import model", command=importModel)
buttonImport.pack()


root.mainloop()


