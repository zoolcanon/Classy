import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Initialize the main window
root = tk.Tk()
root.title("Model Generalization")
root.geometry("1500x800")
root.configure(bg='lightgray')


root.lift()
root.attributes("-topmost",True)
root.after_idle(root.attributes, '-topmost',False)



# Configure grid columns to be centered
for i in range(10):
    root.grid_columnconfigure(i, weight=1)

#Frame for bottom buttons
bottom_frame = tk.Frame(root, bg='gray')
bottom_frame.grid(row=4, column=0, columnspan=10, pady=50)


#These are the fundemental functions
def load_training_data():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files","*.xlsx"),("CSV files","*.csv")])
    if file_path.endswith('.xlsx'):
        df_train = pd.read_excel(file_path)
    else:
        df_train = pd.read_csv(file_path)    
    return df_train
    
    
def load_input_data():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files","*.xlsx"),("CSV files","*.csv")])
    if file_path.endswith('.xlsx'):
        df_input = pd.read_excel(file_path)
    else:
        df_input = pd.read_csv(file_path)
    return df_input
    
#GUI Functions
def on_load_training_data():
    global df_train
    df_train = load_training_data()
    text_train.delete(1.0, tk.END) #when did text_train is defined?
    text_train.insert(tk.END, df_train.head().to_string())
    text_train.tag_add("center", "1.0", "end")


def on_load_input_data():
    global df_input
    df_input = load_input_data()
    text_input.delete(1.0, tk.END)
    text_input.insert(tk.END, df_input.head().to_string())
    text_input.tag_add("center", "1.0", "end")
    enable_buttons_after_loading_data()  # Enable buttons after loading data


# Enable buttons after data is loaded
def enable_buttons_after_loading_data():
    select_button.config(state=tk.NORMAL)
    select_button_advanced.config(state=tk.NORMAL)
    c_button.config(state=tk.NORMAL)
    plot_button.config(state=tk.NORMAL)

# STYLE Button FONTS
button_font = ("Georgia", 11)

# Initialize the results label globally with improved styling
results_label = tk.Label(root, text="", justify="center", anchor="w", wraplength=1000, bg='lightgray', font=("Georgia", 15), borderwidth=5, relief="raised", padx=10, pady=10)


#Button to load training data
load_train_button = tk.Button(root, text= "Load Training Data",command = on_load_training_data,font=button_font, bd=5, relief='raised')
load_train_button.grid(row=2, column=0, columnspan=10, pady=(10, 0))

#Text for demonstration 
text_train = tk.Text(root, height = 10, width = 170, bd=7, relief='sunken') # bd >> boarder of the load frames
text_train.grid(row=3, column=0, columnspan=10, padx=10, pady=(5, 0), sticky='ew')
text_train.tag_configure("center", justify='center')


#Same but for input data
load_input_button = tk.Button(root, text= "Load Input Data",command = on_load_input_data, font=button_font, bd=5, relief='raised')
load_input_button.grid(row=5, column=0, columnspan=10, pady=(1, 0))

text_input = tk.Text(root, height = 10, width = 170, bd=7, relief='sunken')
text_input.grid(row=6, column=0, columnspan=10, padx=10, pady=(5, 10), sticky='ew')
text_input.tag_configure("center", justify='center')


#Now selecting Features
def select_features():
    global feature_columns, target_column
    target_column = simpledialog.askstring("Input","Enter the target column:")
    if target_column not in df_train.columns:
        messagebox.showerror("Error"f"Column {target_column} does not exist in the training set.")
    c = messagebox.askyesno("Input","Do you want to select all features?")
    if c :
        feature_columns = [col for col in df_train.columns if col != target_column]
    else:
        feature_columns = simpledialog.askstring("Input", "Enter feature columns seperated by commas: NOTE*** Entry should percise and case sensitive.")
        feature_columns = [col.strip() for col in feature_columns.split(",")]
   
    check_and_encode_columns()
    train_button.config(state=tk.NORMAL)  # Enable train button after selecting features


#Button to select features and target column
select_button = tk.Button(root, text = "Select Features and Target",command = select_features,font=button_font, bd=5, relief='raised')
select_button.grid(row=11, column=8, padx=5,pady=5, sticky='ew')
select_button.config(state=tk.DISABLED)  # Disable select button initially


def select_features_advanced():
    global numerical_features, categorical_features, target_column, feature_columns
    target_column = simpledialog.askstring("Input","Enter the target column")
    #Error handling
    if target_column not in df_train.columns:
        messagebox.showerror("Error"f"Column {target_column} does not exist in the training set.")
        print("DOES NOT EXIST 128")
        return
    numerical_features = simpledialog.askstring("Input","Enter Numrical Features columns seperated by commas: ")
    numerical_features = [col.strip() for col in numerical_features.split(",")]
    for col in numerical_features:
        if col not in df_train.columns:
            messagebox.showerror("Error"f"Column {col} does not exist in the training set.")
            return
        
    categorical_features = simpledialog.askstring("Input","Enter Categorical Features columns seperated by commas: ")
    categorical_features = [col.strip() for col in categorical_features.split(",")]
    for col in categorical_features:
        if col not in df_train.columns:
            messagebox.showerror("Error"f"Column {col} does not exist in the training set.")
            
            
    #Combining Feature Sets        
    feature_columns = numerical_features + categorical_features
    
    #Encding categorical features
    check_and_encode_columns_advanced()
    train_button.config(state=tk.NORMAL)  # Enable train button after selecting features

             
        
#Specific use case for advanced users
def check_and_encode_columns_advanced():
    global df_train, df_input, encoders, feature_columns
    encoders = {}
    for col in numerical_features + categorical_features + [target_column]:
        if col not in df_train.columns:
            messagebox.showerror("Error",f"Column {col} does not exist in the training data")
            return
        if col not in df_input.columns and col in numerical_features + categorical_features:
            messagebox.showerror("Error",f"Column {col} does not exist in the training data")
            return
    
    
#Encoding catogorical columns
    for col in categorical_features:
        if df_train[col].dtype == 'object':
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col])
            df_input[col] = le.transform(df_input[col])
            encoders[col] = le
    if df_train[target_column].dtype == 'object':
        le = LabelEncoder()
        df_train[target_column] = le.fit_transform(df_train[target_column])
        encoders[target_column] = le
    
    messagebox.showinfo("Info","Columns checked and encoded if needed. You may proceed to train the model.")
    

#Adding the advanced select_feature button
select_button_advanced = tk.Button(root,text = "Select Features (Advanced)",command = select_features_advanced,font=button_font, bd=5, relief='raised')
select_button_advanced.grid(row=11, column=7, padx=5,pady=5, sticky='ew')
select_button_advanced.config(state=tk.DISABLED)  # Disable advanced select button initially



#Cleaning Data
def clean_data():
    global df_train, df_input
    #Numerical columns
    for col in df_train.select_dtypes(include = ['number']).columns:
        df_train[col].fillna(df_train[col].median, inplace = True) #in the training model
        if col in df_input.columns: #in case there was underlapping
            df_input[col].fillna(df_input[col].median, inplace = True)
    
    #Categorical columns
    for col in df_train.select_dtypes(include=['object']).columns:
        df_train[col].fillna(df_train[col].mode()[0], inplace= True)
        if col in df_input.columns:
            df_input[col].fillna(df_input[col].mode()[0], inplace = True)
    messagebox.showinfo("Info","Data cleaned succesfully, Missing Values have been handled")
    
    
#Clean data Button    ############### NO NEED ######################

# clean_data_button = tk.Button(root,text= "Naive Clean",command = clean_data)
# clean_data_button.grid(row=5, column=4, padx=5,pady=5, sticky='ew')



def check_and_encode_columns():
    global df_train, df_input, encoders
    encoders = {}
    
    #for checking---
    for col in feature_columns + [target_column] :
        if col not in df_train.columns:
            messagebox.showerror("Error",f"Column'{col}' does not exist in the training data")
            return
            

    for col in feature_columns:
        if df_train[col].dtype == 'object':
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col])
            df_input[col] = le.transform(df_input[col])
            encoders[col] = le
    if df_train[target_column].dtype == 'object':
        le = LabelEncoder()
        df_train[target_column] = le.fit_transform(df_train[target_column])
        encoders[target_column] = le
    messagebox.showinfo("Info","Columns checked and encoded if needed. You may proceed to train the model.")
    


#Predicting and Saving report
def predict_input_data():
    global df_input, feature_columns, target_column, train_model
    X_input = df_input[feature_columns]
    predictions = train_model.predict(X_input)
    if target_column in encoders:
        predictions = encoders[target_column].inverse_transform(predictions)
    df_input[target_column] = predictions
    for col in feature_columns:
        if col in encoders:
            df_input[col] = encoders[col].inverse_transform(df_input[col])
    save_file()
    


#Training the model
def train_model(): ############### HERE ########################
    global df_train, feature_columns, target_column

    # Check if features and target column are selected 
    try:
        if not feature_columns or not target_column:
            messagebox.showerror("Error", "Please select features and target column first.")
            return
    except NameError:
        messagebox.showerror("Error", "Please select features and target column first.")
        return
    
    X = df_train[feature_columns]
    y = df_train[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    model_pipeline = Pipeline(steps=[('scalar',StandardScaler()), ('classifier', RandomForestClassifier(random_state = 42))])
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)*100
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test,y_pred)
    
    results_text = f"Accuracy: %{accuracy}\n\nClassification Report: \n{report}\n\nConfusion Matrix:\n{conf_matrix}"
    messagebox.showinfo("Model Results",f"Accuracy: %{accuracy}\n\nClassification Report: \n{report}\n\nConfusion Matrix:\n{conf_matrix}")

    # Display the results label
    results_label.config(text=results_text)
    results_label.grid(row=0, column=0, columnspan=10, padx=5, pady=30)


    #function to hide the buttons
    hide_buttons_after_train()


    # Display retry & predict button
    retry_button.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
    predict_button.grid(row=0, column=9, padx=5, pady=5, sticky='ew')

    global train_model
    train_model = model_pipeline

# Hide buttons after training the model
def hide_buttons_after_train():
    load_train_button.grid_remove()
    load_input_button.grid_remove()
    text_train.grid_remove()
    text_input.grid_remove()
    select_button.grid_remove()
    select_button_advanced.grid_remove()
    c_button.grid_remove()
    plot_button.grid_remove()
    train_button.grid_remove()
   
# Retry function
def retry():
    global df_train, df_input
    df_train = None
    df_input = None
    feature_columns = []
    target_column = None
    text_train.delete(1.0, tk.END)
    text_input.delete(1.0, tk.END)
    results_label.config(text="")
    results_label.grid_remove()


    retry_button.grid_remove()
    predict_button.grid_remove()

    restore_initial_buttons()#>>Enable buttons initially
    disable_buttons_initially()#>>Disable buttons initially 

# Restore the initial buttons
def restore_initial_buttons():
    load_train_button.grid(row=2, column=0, columnspan=10, pady=(10, 0))  # Ensure load train button is visible again
    load_input_button.grid(row=5, column=0, columnspan=10, pady=(1, 0))  # Ensure load input button is visible again
    text_train.grid(row=3, column=0, columnspan=10, padx=10, pady=(5, 0), sticky='ew')  # Ensure train data text field is visible again
    text_input.grid(row=6, column=0, columnspan=10, padx=10, pady=(5, 10), sticky='ew')  # Ensure input data text field is visible again
    select_button.grid(row=11, column=8, padx=5, pady=5, sticky='ew')  # Ensure select features button is visible again
    select_button_advanced.grid(row=11, column=7, padx=5, pady=5, sticky='ew')  # Ensure select advanced features button is visible again
    c_button.grid(row=11, column=0, padx=5, pady=5, sticky='ew')  # Ensure clean data button is visible again
    plot_button.grid(row=11, column=1, padx=5, pady=5, sticky='ew')  # Ensure plot button is visible again
    train_button.grid(row=11, column=9, padx=5, pady=5, sticky='ew')  # Ensure train button is visible again

# Disable buttons initially
def disable_buttons_initially():
    select_button.config(state=tk.DISABLED)
    select_button_advanced.config(state=tk.DISABLED)
    c_button.config(state=tk.DISABLED)
    plot_button.config(state=tk.DISABLED)
    train_button.config(state=tk.DISABLED)



#Button to train the model
train_button = tk.Button(root, text="Train", command = train_model,font=button_font, bd=5, relief='raised')
train_button.grid(row=11, column=9, padx=5,pady=5, sticky='ew')
train_button.config(state=tk.DISABLED)  # Initial state is disabled


# Button to retry the process
retry_button = tk.Button(root, text="Retry", command=retry, font=button_font, bd=5, relief='raised')
retry_button.grid(row=0, column=0, columnspan=1, padx=5, pady=5, sticky='ew')
retry_button.grid_remove()

#Button to predict and save the result
predict_button = tk.Button(root, text = "Predict & Save", command = predict_input_data,font=button_font, bd=5, relief='raised')
predict_button.grid(row=0, column=9, padx=5, pady=5,sticky='ew')
predict_button.grid_remove()


#Saving File
def save_file():
    global df_input
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes = [("CSV files","*.csv"), ("Excel files","*.xlsx")])
    if file_path.endswith('.xlsx'):
        df_input.to_excel(file_path, index = False)
    else:
        df_input.to_csv(file_path, index = False)
    messagebox.showinfo("Success",f"File saved to {file_path}")


# Function to plot data
def plot_data():
 
    if df_train is None:
        messagebox.showerror("Error","Please Load a CSV or Excel file first!")
        return
        
    #input window
    input_window = tk.Toplevel(root)
    input_window.title("Select Columns")
    
    #input Frame
    input_frame = tk.Frame(input_window)
    input_frame.pack(pady = 10, padx = 10)
    
    #labels
    feature_column_label = tk.Label(input_frame, text =  "Feature Column: ")
    feature_column_label.grid(row = 0, column = 0, padx= 5, pady=5)
    target_column_label =tk.Label(input_frame, text =  "Target Column: ")
    target_column_label.grid(row = 0, column = 1, padx= 5, pady=5)
    
    

    
    
    # Feature & Target Combo boxes
    feature_column_textvariable = tk.StringVar()
    feature_column_combo = ttk.Combobox(input_frame, textvariable = feature_column_textvariable)
    feature_column_combo['values'] = tuple(df_train.columns)
    feature_column_combo.grid(row =1 , column = 0, padx = 5, pady = 5)
    feature_column_combo.current(0)
    
    
   
    target_column_textvariable = tk.StringVar()
    target_column_combo = ttk.Combobox(input_frame, textvariable = target_column_textvariable)
    target_column_combo['values'] = tuple(df_train.columns)
    target_column_combo.grid(row =1 , column = 1, padx = 5, pady = 5)
    target_column_combo.current(0)
    # 
    # #Entries
    # feature_column_entry = tk.Entry(input_frame)
    # feature_column_entry.grid(row = 1, column =0 , padx = 5, pady = 5)
    # target_column_entry = tk.Entry(input_frame)
    # target_column_entry.grid(row = 1, column = 1, padx = 5, pady = 5)
    
    # plot type selection
    plot_type_label = tk.Label(input_frame, text = "Plot Type: ")
    plot_type_label.grid(row = 2, column = 0,padx= 5, pady =  5)
    plot_type = tk.StringVar()
    plot_type_combo = ttk.Combobox(input_frame, textvariable = plot_type)
    plot_type_combo['values'] = ("Box Plot", 'Scatter Plot')
    plot_type_combo.grid(row =2 , column = 1, padx = 5, pady = 5)
    plot_type_combo.current(0)
    def create_plot():
        feature_column = feature_column_textvariable.get()
        print(feature_column)
        target_column = target_column_textvariable.get()
        selected_plot_type = plot_type.get()
        # if feature_column not in extract_word(str(df_train.columns)) or target_column not in extract_word(str(df_train.columns)):
        #     messagebox.showerror("Error","Feature Column or Target Column does not exist in the dataframe")
        #     return
        #  Clearing previous plots
        for widget in plot_frame.winfo_children():
            widget.destroy()
        fig,ax = plt.subplots(figsize = (10,6))
        if selected_plot_type == 'Box Plot':
            df_train.boxplot(column = feature_column, by = target_column, ax =ax)
        elif selected_plot_type == 'Scatter Plot':
            ax.scatter(df_train[feature_column], df_train[target_column])
            ax.set_xlabel(feature_column)
            ax.set_ylabel(target_column)
            ax.set_xlim([df_train[feature_column].min()*0.9,df_train[feature_column].max()*1.1])
            ax.set_ylim([df_train[target_column].min()*0.9,df_train[target_column].max()*1.1])
            print("----")
            print(ax.get_xlim())
            # Titeling
        ax.set_title(f'{selected_plot_type}: {feature_column} vs {target_column}')
        # plt.sub("")
            
    
    
        canvas = FigureCanvasTkAgg(fig, master = plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = 1)
        print("plot displayed")
            
    # plot frame
    plot_frame = tk.Frame(input_window)
    plot_frame.pack(fill = tk.BOTH, expand = 1)
    
    create_plot_button = tk.Button(input_frame,text = "Create Plot", command = create_plot)
    print("Line 99")
    create_plot_button.grid(row =3 , column = 1, padx = 5, pady = 5)
    print("Line101")


def clean_data():
    df = df_train
    if df is None:
        messagebox.showerror("Error","Please load an excel or CSV file first!")
    def get_cleaning_methods(col_type):
        if col_type == 'object':
            return ['Mode', 'Remove','Replace']
        else:
            return ['Mean','Median','Mode','Remove','Replace']
    def apply_cleaning():
        selected_column = column_combo.get()
        selected_method = method_combo.get()
        replace_from = rf_entry.get()
        replace_to = rt_entry.get()
        
        #i don't think it's necessary
        if selected_column not in df.columns:
            messagebox.showerror("Error",f"Column {selected_column} not found in the data")
            return
        col_type = df[selected_column].dtype
        print(f"Selected column {selected_column} with type {col_type} and method: {selected_method}")
        messagebox.showinfo("Process",f"Selected column {selected_column} with type {col_type} and method: {selected_method}")
        if selected_method == 'Mean' and col_type != 'object':
            df[selected_column].fillna(df[selected_column].mean(), inplace = True)
        elif selected_method == 'Median' and col_type != 'object':
            df[selected_column].fillna(df[selected_column].median(), inplace = True)
        elif selected_method == 'Mode' and col_type != 'object':
            df[selected_column].fillna(df[selected_column].mode()[0], inplace = True)
        elif selected_method == 'Remove':
            df.dropna(subset = [selected_column],inplace = True)
        elif selected_method == 'Replace':
            if replace_from == '' or replace_to == '':
                messagebox.showerror("Error","Please specify both values for replacement")
                return
            if col_type != 'object':
                try:
                    replace_from = float(replace_from)
                except ValueError:
                    messagebox.showerror("Error",f"Value {replace_from} is not a valid number")
                    return
                try:
                    replace_to = float(replace_to)
                except ValueError:
                    messagebox.showerror("Error",f"Value {replace_to} is not a valid number")
                    return
            df[selected_column].replace(replace_from,replace_to,inplace=True)
        else:
            messagebox.showerror("Error","Invalid cleaning method")
            return


        
        update_description()
        update_info()
    def update_description():
        nd = df.describe(include = "all").to_string()
        text_widget.config(state = tk.NORMAL)
        text_widget.delete(1.0,tk.END)
        text_widget.insert(tk.END,nd)
        text_widget.config(state = tk.DISABLED)
        
    
    
    def update_info():
        buffer = io.StringIO()
        df.info(buf = buffer)
        info = buffer.getvalue()
        new_info = (f"Number of rows: {len(df)}\n"f"Number of columns: {df.shape[1]}\n""\nColumn Info: {info}\n"f"\nMissing Values:\n {df.isnull().sum().to_string()}")
     
        info_widget.config(state=tk.NORMAL)
        info_widget.delete(1.0,tk.END)
        info_widget.insert(tk.END,new_info)
        info_widget.config(state = tk.DISABLED)
        
        
        
        
        
    #toplevel window
    clean_window = tk.Toplevel(root)
    clean_window.title("Data Cleaner")
    
    
    #inputframe
    input_frame = tk.Frame(clean_window)
    input_frame.pack(pady= 10, padx=10)
    

    description_frame = tk.Frame(input_frame)
    description_frame.grid(row = 0,column = 0, columnspan = 2,padx=5,pady=5)
    
    canvas = tk.Canvas(description_frame, height =150)
    scrollbar_y = tk.Scrollbar(description_frame,orient = "vertical",command = canvas.yview)
    scrollbar_x=tk.Scrollbar(description_frame,orient = "horizontal",command = canvas.xview)
    scrollable_frame = tk.Frame(canvas)
    
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion = canvas.bbox("all")))
    canvas.create_window((0,0), window = scrollable_frame,anchor="nw")
    canvas.configure(yscrollcommand = scrollbar_y.set,xscrollcommand = scrollbar_x)
    df_description = df.describe(include = "all").to_string()
    
    text_widget = tk.Text(scrollable_frame, height=30,width =200, wrap = "none")
    text_widget.insert(tk.END,df_description)
    text_widget.config(state = tk.DISABLED)
    text_widget.pack()
    
    canvas.pack(side = "left",fill = "both",expand = True)
    scrollbar_y.pack(side = "right",fill = "y")
    scrollbar_x.pack(side = "bottom",fill = "x")
    
    #info text widget
    info_widget = tk.Text(input_frame,height = 10,width=80,wrap="none")
    info_widget.grid(row = 1, column = 0,columnspan = 2,padx = 5,pady = 5)
    update_info() #Calling the function
    
    
        
    #column selection
    select_column_label = tk.Label(input_frame, text = "Select Column: ")
    select_column_label.grid(row = 2, column = 0, padx=5,pady=5)
        
    column_combo = ttk.Combobox(input_frame, values = list(df.columns))
    column_combo.grid(row = 2, column = 1, padx=5,pady=5)
    
    #method selection
    select_method_label = tk.Label(input_frame, text = "Select Method: ")
    select_method_label.grid(row = 3, column = 0, padx=5,pady=5)
    method_combo = ttk.Combobox(input_frame, values = list(df.columns))
    method_combo.grid(row = 3, column = 1, padx=5,pady=5)
    method_combo.grid_remove()
        
    #replace input fields
    rf_label = tk.Label(input_frame, text = "Replace from: ")
    rf_label.grid(row = 4, column = 0, padx=5,pady=5)
    rf_entry = tk.Entry(input_frame)
    rf_entry.grid(row=4,column =1, padx=5,pady=5)
        
    rt_label = tk.Label(input_frame, text = "Replace to: ")
    rt_label.grid(row = 5, column = 0, padx=5,pady=5)
    rt_entry = tk.Entry(input_frame)
    rt_entry.grid(row=5,column =1, padx=5,pady=5)
        
    #hide replace initially
    rf_label.grid_remove()
    rf_entry.grid_remove()
    rt_label.grid_remove()
    rt_entry.grid_remove()
        
        
    def updated_methods(event):
        method_combo.grid()
        selected_column = column_combo.get()
        if selected_column in df.columns:
            col_type = df[selected_column].dtype
            method_combo['values'] = get_cleaning_methods(col_type)
            method_combo.current(0)
            update_replace_fields()
                
    def update_replace_fields(event = None):
        selected_method = method_combo.get()
        if selected_method == 'Replace':
            rf_label.grid()
            rf_entry.grid()
            rt_label.grid()
            rt_entry.grid()
        else:
            rf_label.grid_remove()
            rf_entry.grid_remove()
            rt_label.grid_remove()
            rt_entry.grid_remove()
        
        
    column_combo.bind("<<ComboboxSelected>>",updated_methods)
    method_combo.bind("<<ComboboxSelected>>",update_replace_fields) # what happened here -- event easy
    
        
    clean_button = tk.Button(input_frame,text="Clean",command = apply_cleaning)
    clean_button.grid(row = 5,columnspan=2,pady=10, sticky='')



#Cleaning button
c_button = tk.Button(root, text = "Clean Data", command = clean_data,font=button_font, bd=5, relief='raised')
c_button.grid(row=11, column=0, padx=5,pady=5, sticky='ew')
c_button.config(state=tk.DISABLED)  # Disable clean data button initially

    
#Creating a frame for the plot
plot_button = tk.Button(root,text = "Plot", command  = plot_data ,font=button_font, bd=5, relief='raised')
plot_button.grid(row=11, column=1, padx=5, pady=5, sticky='ew')
plot_button.config(state=tk.DISABLED)  # Disable plot button initially


disable_buttons_initially()  # Call function to disable buttons initially

#MainLoop
root.mainloop()
