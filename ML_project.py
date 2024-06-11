import shutil
import os
import tkinter
import tksheet
from pandastable import Table, TableModel
from tkinter import BOTH, Frame, filedialog
from tkinter import ttk
from sklearn.linear_model import LinearRegression,LogisticRegression;import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkinter.messagebox
import customtkinter
import pandas as pd
import numpy as np

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.counter_normal = 0
        self.counter_train = 0
        self.counter_test = 0
        self.counter_trainxy = 0
        self.df = None
        #Data
        self.train = None
        self.test = None
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.model = None
        self.predict = None
        # configure window
        self.title("Model Trainer")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (3x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure(0, weight=2)

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label_1 = customtkinter.CTkLabel(self.sidebar_frame, text="Model Modefier", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label_1.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.logo_label_2 = customtkinter.CTkLabel(self.tabview, text="Please, Choose Data", font=customtkinter.CTkFont(size=20, weight="bold") , bg_color="red")
    
        self.logo_label_2.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.input_file)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))

        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))

        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
     
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))

        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create Table
        self.show_table = customtkinter.CTkScrollableFrame(self, label_text="Data")
        self.show_table.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # set default values for left column
        self.sidebar_button_1.configure(text="Choose File")
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def input_file(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.logo_label_2.destroy()
            self.create_table()
            self.df = pd.read_csv(self.file_path)
            self.show_preprosessing_tab()
    
    def show_preprosessing_tab(self):
        self.tabview.add("Preprosessing")
        self.tabview.tab("Preprosessing").grid_columnconfigure(0, weight=1)
        self.pre_label_1 = customtkinter.CTkLabel(self.tabview.tab("Preprosessing"),
                                                  text="Missing Values",
                                                  bg_color="black",
                                                  font=customtkinter.CTkFont(size=12, weight="bold"))
        self.pre_label_1.grid(row=0, column=0,columnspan=2, padx=20, pady=(10, 10))
        self.pre_handle_nulls = customtkinter.CTkEntry(self.tabview.tab("Preprosessing"),
                                                   placeholder_text="Type to convert to NAN")
        self.pre_handle_nulls.grid(row=1,column=0,padx=10, pady=(10, 10))
        self.pre_handle_nulls_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                     text = "Convert",
                                                     command=self.pre_handle_nulls_button_event)
        self.pre_handle_nulls_button.grid(row=1, column=1, padx=10, pady=(10, 10))
        self.pre_slider_1 = customtkinter.CTkSlider(self.tabview.tab("Preprosessing"),
                                                    from_=0,to=100,
                                                    command=self.pre_slider_1_event,
                                                    number_of_steps=20)
        self.pre_slider_1.grid(row=2, column=0,columnspan=2, padx=20, pady=(10, 10))
        self.pre_label_2 = customtkinter.CTkLabel(self.tabview.tab("Preprosessing"),
                                                  text="Test Data 50%",
                                                  font=customtkinter.CTkFont(size=12, weight="bold"))
        self.pre_label_2.grid(row=3, column=0, padx=20, pady=(10, 10))
        self.pre_split_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                text="Split",
                                                command=self.pre_split_button_event)
        self.pre_split_button.grid(row=3, column=1, padx=20, pady=(10, 10))

        self.pre_deletecol_entry = customtkinter.CTkEntry(self.tabview.tab("Preprosessing"),
                                                   placeholder_text="Type Column to Delete")
        self.pre_deletecol_entry.grid(row=4,column=0,padx=10, pady=(10, 10))

        self.pre_deletecol_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                                text="Delete",
                                                                command=self.pre_deletecol_button_event)
        self.pre_deletecol_button.grid(row=4, column=1, padx=20, pady=(10, 10))

        self.pre_missingvalues_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                                text="Handle Missing values",
                                                                command=self.impute_missing_values)
        self.pre_missingvalues_button.grid(row=5, column=0,columnspan=2, padx=20, pady=(10, 10))

        self.pre_scalerscale_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                                text="Scaler Scaling",
                                                                command=self.pre_scalerscale_button_event)
        self.pre_scalerscale_button.grid(row=6, column=0, padx=20, pady=(10, 10))

        self.pre_minmaxscale_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                                text="MinMax Scaling",
                                                                command=self.pre_minmaxscale_button_event)
        self.pre_minmaxscale_button.grid(row=6, column=1, padx=20, pady=(10, 10))

        self.pre_laabelencode_entry = customtkinter.CTkEntry(self.tabview.tab("Preprosessing"),
                                                   placeholder_text="Type Column Name")
        self.pre_laabelencode_entry.grid(row=7,column=0,padx=10, pady=(10, 10))

        self.pre_labelencode_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                                text="Label Encode",
                                                                command=self.pre_labelencode_button_event)
        self.pre_labelencode_button.grid(row=7, column=1, padx=20, pady=(10, 10))

        self.pre_oneencode_entry = customtkinter.CTkEntry(self.tabview.tab("Preprosessing"),
                                                   placeholder_text="Type Column Name")
        self.pre_oneencode_entry.grid(row=8,column=0,padx=10, pady=(10, 10))

        self.pre_oneencode_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                                text="One Encode",
                                                                command=self.pre_oneencode_button_event)
        self.pre_oneencode_button.grid(row=8, column=1, padx=20, pady=(10, 10))

        self.pre_ytrain_entry = customtkinter.CTkEntry(self.tabview.tab("Preprosessing"),
                                                   placeholder_text="Type YTrain Column")
        self.pre_ytrain_entry.grid(row=9,column=0,padx=10, pady=(10, 10))

        self.pre_ytrain_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                                text="Select Y-Train",
                                                                command=self.pre_ytrain_button_event)
        self.pre_ytrain_button.grid(row=9, column=1, padx=20, pady=(10, 10))

        self.pre_balancing_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                                text="Balance Data",
                                                                command=self.pre_balancing_button_event)
        self.pre_balancing_button.grid(row=10, column=0,columnspan=2, padx=20, pady=(10, 10))

        self.pre_done_button = customtkinter.CTkButton(self.tabview.tab("Preprosessing"),
                                                  text="Done",
                                                  command=self.pre_done_button_event)
        self.pre_done_button.grid(row=12, column=0,columnspan=2, padx=20, pady=(10, 10))

    def pre_handle_nulls_button_event(self):
        self.df.replace(str(self.pre_handle_nulls.get()),np.nan,inplace=True)
        self.save_load_normal()
    
    def pre_split_button_event(self):
        self.train,self.test = train_test_split(self.df,test_size=self.pre_slider_1.get()/100,random_state=42)
        self.save_load_train()

    def pre_labelencode_button_event(self):
        col = self.pre_laabelencode_entry.get()
        le = LabelEncoder()
        self.train[col] = le.fit_transform(self.train[col])
        self.test[col] = le.transform(self.test[col])
        self.save_load_train()

    def pre_oneencode_button_event(self):
        col = self.pre_oneencode_entry.get()
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_train = one_hot_encoder.fit_transform(self.train[[col]])
        one_hot_train_df = pd.DataFrame(one_hot_train, columns= one_hot_encoder.get_feature_names_out([col]), index=self.train.index)
        self.train = pd.concat([self.train,one_hot_train_df],axis=1)
        self.train.drop(col,inplace=True,axis=1)
        one_hot_test = one_hot_encoder.transform(self.test[[col]])
        one_hot_test_df = pd.DataFrame(one_hot_test, columns= one_hot_encoder.get_feature_names_out([col]), index=self.test.index)
        self.test = pd.concat([self.test,one_hot_test_df],axis=1)
        self.test.drop(col,inplace=True,axis=1)
        self.save_load_train()

    def pre_scalerscale_button_event(self):
        numerical_cols = self.train.select_dtypes(include=['number']).columns
        scaler = StandardScaler()
        self.train[numerical_cols] = scaler.fit_transform(self.train[numerical_cols])
        self.test[numerical_cols] = scaler.transform(self.test[numerical_cols])
        self.save_load_train()

    def pre_minmaxscale_button_event(self):
        numerical_cols = self.train.select_dtypes(include=['number']).columns
        minmax = MinMaxScaler()
        self.train[numerical_cols] = minmax.fit_transform(self.train[numerical_cols])
        self.test[numerical_cols] = minmax.transform(self.test[numerical_cols])
        self.save_load_train()

    def pre_balancing_button_event(self):
        smt = SMOTETomek(random_state=42)
        self.xtrain,self.ytrain = smt.fit_resample(self.xtrain,self.ytrain)
        self.save_load_trainxy()

    def pre_ytrain_button_event(self):
        value = self.pre_ytrain_entry.get()
        self.xtrain = self.train.drop(columns=value)
        self.ytrain = self.train[value]
        self.xtest = self.test.drop(columns=value)
        self.ytest = self.test[value]
        self.save_load_trainxy()

    def pre_deletecol_button_event(self):
        value = self.pre_deletecol_entry.get()
        self.train = self.train.drop(columns=value)
        self.test = self.test.drop(columns=value)
        self.save_load_train()

    def pre_done_button_event(self):
        self.show_Algorithm_tab()

    def show_Algorithm_tab(self):
        #Tab 2(Algorithm)
        self.tabview.add("Algorithm")
        self.tabview.tab("Algorithm").grid_columnconfigure(0, weight=1) #Show Preprocessing Tab

        self.optionmenu_2_1 = customtkinter.CTkOptionMenu(self.tabview.tab("Algorithm"), dynamic_resizing=False,
                                                        values=["Linear Regression", "Logistic Regression","Support vector machine"])
        self.optionmenu_2_1.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Algorithm"), text="Train",command=self.alg_train_button_event)
        self.string_input_button.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.score = customtkinter.CTkLabel(self.tabview.tab("Algorithm"),
                                            text='',
                                            font=customtkinter.CTkFont(size=16, weight="bold"))
        self.score.grid(row=6,
                        column=0,
                        padx=20,
                        pady=(20, 10))

        #set default values for Tab 2
        self.optionmenu_2_1.set("Algorithm")

    def alg_train_button_event(self):
        choice = self.optionmenu_2_1.get()
        if (choice == "Linear Regression"):

            regressor = LinearRegression(positive=False)
            regressor.fit(self.xtrain, self.ytrain)
            y_pred_train = regressor.predict(self.xtrain)
            accuracy = accuracy_score(self.ytest, y_pred_train)
            self.score.configure(text=f"Accuracy: {accuracy}")
        elif (choice == "Logistic Regression"):
            lg = LogisticRegression()
            lg.fit(self.xtrain,self.ytrain)
            y_pred = lg.predict(self.xtest)
            accuracy = accuracy_score(self.ytest, y_pred)
            self.score.configure(text=f"Accuracy: {accuracy}")
        elif(choice == "Support vector machine"):
            self.score.configure(text=f"Score: {self.score}%")
            clf = SVC(kernel="linear")
            clf.fit(self.xtrain,self.ytrain)
            y_pred = clf.predict(self.xtest)
            accuracy = accuracy_score(self.ytest, y_pred)
            self.score.configure(text=f"Accuracy: {accuracy}")
    
    def create_table(self):
        self.table = Table(self.show_table,showstatusbar=True)
        self.table.grid(row=0, column=0)
        self.table.importCSV(self.file_path)
        self.table.show()

    def pre_slider_1_event(self,value):
        self.pre_label_2.configure(text="Test data "+str(value)+"%")

    def save_load_normal(self):
        self.counter_normal = self.counter_normal + 1
        self.df.to_csv(f"Tempo/normal_{self.counter_normal}", index=False)
        self.file_path = f"Tempo/normal_{self.counter_normal}"
        self.df = pd.read_csv(self.file_path)
        self.table.importCSV(self.file_path)
        self.table.show()
    def save_load_train(self):
        self.counter_train = self.counter_train + 1
        self.counter_test = self.counter_test + 1
        self.train.to_csv(f"Tempo/train_{self.counter_train}", index=False)
        self.test.to_csv(f"Tempo/test_{self.counter_test}", index=False)
        self.file_path = f"Tempo/train_{self.counter_train}"
        self.train = pd.read_csv(self.file_path)
        self.test = pd.read_csv(f"Tempo/test_{self.counter_test}")
        self.table.importCSV(self.file_path)
        self.table.show()
    def save_load_trainxy(self):
        self.counter_trainxy = self.counter_trainxy + 1
        self.xtrain.to_csv(f"Tempo/xtrain_{self.counter_trainxy}", index=False)
        self.xtest.to_csv(f"Tempo/xtest_{self.counter_trainxy}", index=False)
        self.ytrain.to_csv(f"Tempo/ytrain_{self.counter_trainxy}", index=False)
        self.ytest.to_csv(f"Tempo/ytest_{self.counter_trainxy}", index=False)
        self.file_path = f"Tempo/xtrain_{self.counter_trainxy}"
        self.xtrain = pd.read_csv(self.file_path)
        self.xtest = pd.read_csv(f"Tempo/xtest_{self.counter_trainxy}")
        self.ytrain = pd.read_csv(f"Tempo/ytrain_{self.counter_trainxy}")
        self.ytest = pd.read_csv(f"Tempo/ytest_{self.counter_trainxy}")
        self.table.importCSV(self.file_path)
        self.table.show()

    def diplay_nulls(self):
        print(self.train.isnull().sum(axi=1).sort_values(ascending=False))

    def impute_missing_values(self):
        numerical_cols_train = self.train.select_dtypes(include=['number']).columns
        categorical_cols_train = self.train.select_dtypes(include=['object', 'category']).columns
        numerical_cols_test = self.test.select_dtypes(include=['number']).columns
        categorical_cols_test = self.test.select_dtypes(include=['object', 'category']).columns

        median_imputer = SimpleImputer(strategy='median')
        mode_imputer = SimpleImputer(strategy='most_frequent')
        
        self.train[numerical_cols_train] = median_imputer.fit_transform(self.train[numerical_cols_train])
        self.train[categorical_cols_train] = mode_imputer.fit_transform(self.train[categorical_cols_train])

        self.test[numerical_cols_test] = median_imputer.fit_transform(self.test[numerical_cols_test])
        self.test[categorical_cols_test] = mode_imputer.fit_transform(self.test[categorical_cols_test])

        self.save_load_train()

if __name__ == "__main__":
    def on_closing():
        shutil.rmtree("Tempo")
        os.mkdir("Tempo")
    app = App()
    app.protocol("WM_DELETE_WINDOW", on_closing())
    app.mainloop()