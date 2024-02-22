# Import necessary libraries and modules
import streamlit as st
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set the title and add an image to the app
st.title("*Machine Learning App*")
image = Image.open("img.png")
st.image(image, use_column_width=True)

# Define the main function
def main():

    # Create a sidebar with options for EDA, Visualization, and Models
    option = st.sidebar.selectbox("Select option", ('EDA', 'Visualisation', 'Model'))

    # Exploratory Data Analysis (EDA) Section
    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')

        # Allow the user to upload a file (csv or xlsx)
        file = st.file_uploader("Choose Your File", type=['csv', 'xlsx'])
        if file:
            st.success("Successfully Uploaded")
            data = pd.read_csv(file)

        # Display various aspects of the dataset based on user checkboxes
        dataset_checkbox = st.checkbox("Display Dataset")
        if dataset_checkbox:
            if file is not None:
                st.dataframe(data)
            else:
                st.error("Upload A File")

        shape_checkbox = st.checkbox("Display Shape")
        if shape_checkbox:
            if file is not None:
                st.write("Shape of Your Dataset is:", data.shape)
            else:
                st.error("Upload A File")

        column_checkbox = st.checkbox("Display Columns")
        if column_checkbox:
            if file is not None:
                st.write(data.columns)
            else:
                st.error("Upload A File")

        multiple_columns_checkbox = st.checkbox("Select Multiple Columns")
        if multiple_columns_checkbox:
            if file is not None:
                column_list = st.multiselect("Choose Columns", options=data.columns)
                column_dataframe = pd.DataFrame(data[column_list])
                st.dataframe(column_dataframe)
            else:
                st.error("Upload A File")

        summary_checkbox = st.checkbox("Display Summary")
        if summary_checkbox:
            if file is not None:
                st.write(data.describe().T)
            else:
                st.error("Upload A File")

        null_values_checkbox = st.checkbox("Display Null Values")
        if null_values_checkbox:
            if file is not None:
                st.write(data.isnull().sum())
            else:
                st.error("Upload A File")

        datatypes_checkbox = st.checkbox("Display Datatypes")
        if datatypes_checkbox:
            if file is not None:
                st.dataframe(data.dtypes, column_config={'': 'Column Name', '0': 'DataTypes'}, use_container_width=True)
            else:
                st.error("Upload A File")

        correlation_checkbox = st.checkbox("Display Correlation Matrix")
        if correlation_checkbox:
            if file is not None:
                st.write(data.corr())
            else:
                st.error("Upload A File")

    # Data Visualization Section            
    elif option=='Visualisation':
        st.subheader("Data Visualisation")

        # Allow the user to upload a file (csv or xlsx)
        file=st.file_uploader("Choose Your File",type=['csv','xlsx'])
        if file:
            st.success("Successfully Uploaded")
            data=pd.read_csv(file)

        # Display various aspects of the dataset and visualizations based on user checkboxes
        dataset_checkbox=st.checkbox("Display Dataset")
        if dataset_checkbox==True:
            if file is not None:
                st.dataframe(data)
            else:
                st.error("Upload A File")
        
        column_checkbox=st.checkbox("Display Columns")
        if column_checkbox==True:
            if file is not None:
                st.write(data.columns)
            else:
                st.error("Upload A File")

        heatmap_checkbox=st.checkbox("Display Heatmap")
        if heatmap_checkbox:
            if file is not None:
                column_list=st.multiselect("Choose Columns",options=data.columns)
                if column_list:
                    heat_data=pd.DataFrame(data[column_list])
                    st.dataframe(heat_data)
                    shape=heat_data.corr().shape[0]
                    fig=plt.figure(figsize=(shape,shape))
                    st.write(sns.heatmap(heat_data.corr(),annot=True,square=True))
                    st.pyplot(fig,use_container_width=True)
                else:
                    pass
            else:
                st.error("Upload A File")

    # Machine Learning Model Section
    else:
        st.subheader("Models")

        # Allow the user to upload a file (csv or xlsx)
        file=st.file_uploader("Choose Your File",type=['csv','xlsx'])
        if file:
            st.success("Successfully Uploaded")
            data=pd.read_csv(file)

        # Allow the user to select multiple columns and choose a classifier
        multiple_columns_checkbox=st.checkbox("Select Multiple Columns")
        if multiple_columns_checkbox:
            if file is not None:
                column_list=st.multiselect("Choose Columns",options=data.columns)
                new_data=data[column_list]
                st.dataframe(new_data)
                            
            else:
                st.error("Upload A File")

            # Set up classifiers (KNN, SVC, Logistic Regression) based on user choices
            x=new_data.iloc[:,0:-1]
            y=new_data.iloc[:,-1]
            seed=st.sidebar.slider("Seed",1,200)
            classifier_name=st.sidebar.selectbox("Select Classifier",('KNN','SVC','Logistic Regression'))
            
            # Define functions to set parameters and create the classifier
            def parameters(classifier):
                param=dict()
                if classifier=='KNN':
                    K=st.sidebar.slider('K',1,15)
                    param['K']=K
                else:
                    C=st.sidebar.slider('C',0.01,15.0)
                    param['C']=C
                return param
            
            params=parameters(classifier_name)

            def classifier(classifier,params):
                model=None
                if classifier=='KNN':
                    model=KNeighborsClassifier(n_neighbors=params['K'])
                elif classifier=='SVC':
                    model=SVC(C=params['C'])
                else:
                    model=LogisticRegression()
                return model
            
            # Train the model, make predictions, and display results
            model=classifier(classifier_name,params)
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=seed)
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            st.write('Prediction :',y_pred)
            st.write('Classifier Name :',classifier_name)
            st.write('Accuracy Score :{}%'.format(accuracy_score(y_test,y_pred)*100))

# Run the main function
if __name__=="__main":
    main()
