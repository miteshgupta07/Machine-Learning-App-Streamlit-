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

st.title("*Machine Learning App*")
image=Image.open("E:\Data Science\Streamlit\Images\WebApp-2.png")
st.image(image,use_column_width=True)

def main():
    option=st.sidebar.selectbox("Select option",('EDA','Visualisation','Model'))
    if option=='EDA':
        st.subheader('Exploratory Data Analysis')
        
        file=st.file_uploader("Choose Your File",type=['csv','xlsx'])
        if file:
            st.success("Successfully Uploaded")
            data=pd.read_csv(file)

        dataset_checkbox=st.checkbox("Display Dataset")
        if dataset_checkbox==True:
            if file is not None:
                st.dataframe(data)
            else:
                st.error("Upload A File")

        shape_checkbox=st.checkbox("Display Shape")
        if shape_checkbox==True:
            if file is not None:
                st.write("Shape of Your Dataset is :",data.shape)
            else:
                st.error("Upload A File")
            
        column_checkbox=st.checkbox("Display Columns")
        if column_checkbox==True:
            if file is not None:
                st.write(data.columns)
            else:
                st.error("Upload A File")

        multiple_columns_checkbox=st.checkbox("Select Multiple Columns")
        if multiple_columns_checkbox==True:
            if file is not None:
                column_list=st.multiselect("Choose Columns",options=data.columns)
                column_dataframe=pd.DataFrame(data[column_list])
                st.dataframe(column_dataframe)
            else:
                st.error("Upload A File")   
            
        summary_checkbox=st.checkbox("Display Summary")
        if summary_checkbox==True:
            if file is not None:
                st.write(data.describe().T)
            else:
                st.error("Upload A File")
        
        null_values_ckeckbox=st.checkbox("Display Null Value")
        if null_values_ckeckbox:
            if file is not None:
                st.write(data.isnull().sum())
            else:
                st.error("Upload A File")

        datatypes_checkbox=st.checkbox("Display Datatypes")
        if datatypes_checkbox:
            if file is not None:
                st.dataframe(data.dtypes,column_config={'':'Column Name','0':'DataTypes'},use_container_width=True)
            else:
                st.error("Upload A File")

        correlation_checkbox=st.checkbox("Display Correlation")
        if correlation_checkbox:
            if file is not None:
                st.write(data.corr())
            else:
                st.error("Upload A File")
                
    elif option=='Visualisation':
        st.subheader("Data Visualisation")
        file=st.file_uploader("Choose Your File",type=['csv','xlsx'])
        if file:
            st.success("Successfully Uploaded")
            data=pd.read_csv(file)

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

        multiple_columns_checkbox=st.checkbox("Select Multiple Columns")
        if multiple_columns_checkbox==True:
            if file is not None:
                column_list=st.multiselect("Choose Columns",options=data.columns)
                column_dataframe=pd.DataFrame(data[column_list])
                st.dataframe(column_dataframe)
            else:
                st.error("Upload A File")

        heatmap_checkbox=st.checkbox("Display Heatmap")
        if heatmap_checkbox:
            if file is not None:
                shape=data.corr().shape[0]
                fig=plt.figure(figsize=(shape,shape))
                st.write(sns.heatmap(data.corr(),annot=True,square=True))
                st.pyplot(fig,use_container_width=True)
            else:
                st.error("Upload A File")

    else:
        st.subheader("Models")
        file=st.file_uploader("Choose Your File",type=['csv','xlsx'])
        if file:
            st.success("Successfully Uploaded")
            data=pd.read_csv(file)

        multiple_columns_checkbox=st.checkbox("Select Multiple Columns")
        if multiple_columns_checkbox==True:
            if file is not None:
                column_list=st.multiselect("Choose Columns",options=data.columns)
                new_data=data[column_list]
                st.dataframe(new_data)            
            else:
                st.error("Upload A File")

            x=new_data.iloc[:,0:-1]
            y=new_data.iloc[:,-1]
            seed=st.sidebar.slider("Seed",1,200)
            classifier_name=st.sidebar.selectbox("Select Classifier",('KNN','SVC','Logistic Regression'))
            
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
            
            model=classifier(classifier_name,params)
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=seed)
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            st.write('Prediction :',y_pred)
            st.write('Classifier Name :',classifier_name)
            st.write('Accuracy Score :{}%'.format(accuracy_score(y_test,y_pred)*100))

main()