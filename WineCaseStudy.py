import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def ClassificationKNN(Datapath):
    Border = "-"*60
    ####################################
    #Step 1 : Load the Dataset 
    ####################################

    print(Border)
    print("Step 1 : Load the Dataset ")
    print(Border)

    df = pd.read_csv(Datapath)

    print("Initial 5 entries from dataset:")
    print(df.head())

    print(Border)

    print("Last % Records from Dataset : ")
    print(df.tail())

    ####################################
    #Step 2 : Analyze the Dataset 
    ####################################
    
    print(Border)
    print("Step 2 : Analyze the Dataset")
    print(Border)

    df.dropna(inplace=True)
    #The dropna() method removes the rows that contains NULL values. 
    #The dropna() method returns a new DataFrame object unless the inplace parameter is set to True, 
    #in that case the dropna() method does the removing in the original DataFrame instead.

    print("Shape of Dataset : ",df.shape)
    print(Border)

    print("Total Record : ",df.shape[0])
    print("Total Columns : ",df.shape[1])

    print(Border)
    print("Statistical Summary of Dataset : ")
    print(df.describe())

    ########################################################
    #Step 3 : Separate Dependent and Independent Varable
    ########################################################
        
    print(Border)
    print("Step 3 : Separate Dependent and Independent Varable")
    print(Border)

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of Columns : ",X.shape)
    print("Shape of Rows : ",Y.shape)

    print(Border)

    print("List of all input columns : ",X.columns.tolist())
    print(Border)
    print("Output columns : [Class]")

    ########################################################
    #Step 4 : Split the Dataset into Training and Testing 
    ########################################################
        
    print(Border)
    print("Step 4 : Split the Dataset into Training and Testing ")
    print(Border)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state=42,stratify=Y)
    #stratify parameter ensures that the training and testing sets have the same proportion of classes (or labels) 
    # as the original dataset.

    print(Border)
    print("Shape of X_train : ",X_train.shape)
    print("SHape of X_test : ",X_test.shape)
    print("Shapeof Y_train : ",Y_train.shape)
    print("Shape of Y_test : ",Y_test.shape)
    print(Border)

    ########################################################
    #Step 5 : Feature Scaling
    ########################################################
        
    print(Border)
    print("Step 5 : Feature Scaling")
    print(Border)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    print("Feature Scaling Done")
    print(Border)

    # #########################################
    # #Step 6 : Build the Model
    # #########################################

    # print(Border)
    # print("Step 6 : Build the Model")
    # print(Border)

    # model = KNeighborsClassifier(n_neighbors=5)
    # print("Classification model is created")
    # print(Border)

    # #########################################
    # #Step 7 : Train the Model
    # #########################################

    # print(Border)
    # print("Step 7 : Train the Model")
    # print(Border)

    # model = model.fit(X_train_scaled,Y_train)
    # print("Model Trained Successfully")
    # print(Border)

    # #########################################
    # #Step 8 : Test the Model
    # #########################################
 
    # print(Border)
    # print("Step 7 : Test the Model")
    # print(Border)   

    # Y_pred = model.predict(X_test_scaled)

    # accuracy = accuracy_score(Y_test,Y_pred)

    # print("Accuracy score : ",accuracy*100)

    print(Border)
    print("Highper Parameter Tuning")
    print(Border)

    accuracy_scores = []

    K_Values = range(1,21)

    for k in K_Values:
        model = KNeighborsClassifier(n_neighbors=k)
        model = model.fit(X_train_scaled,Y_train)
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test,Y_pred)

        accuracy_scores.append(accuracy*100)

    print("Accuracy Report : ")
    for k,no in zip(K_Values,accuracy_scores):
        print(k,":",no)

    print(Border)

    print(Border)

    print("Graphical Representation : ")
    print(Border)

    plt.figure(figsize=(8,5))

    plt.plot(K_Values,accuracy_scores,marker = "o")
    plt.title("K Values and Accuracy")

    plt.xlabel("Values of K")
    plt.ylabel("Accuracy")

    plt.grid(True)
    plt.xticks(list(K_Values))
    plt.show()

def main():
    ClassificationKNN("WinePredictor.csv")

if __name__ == "__main__":
    main()