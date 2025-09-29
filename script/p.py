import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

class EDA:
    
    def eda(self):

        st.title("Upload and Read Dataset")
        self.uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if self.uploaded_file is not None:  
            st.session_state.df = pd.read_csv(self.uploaded_file)
            self.df = st.session_state.df

            place_holder = st.empty()
            place_holder.success("✅ File Uploaded Successfully!")
            time.sleep(1)
            place_holder.empty()
            
        elif "df" in st.session_state:
            self.df = st.session_state.df
            st.info("Using previously uploaded dataset ✅")

            
            st.dataframe(self.df.head()) 
            
            time.sleep(1.5)
            
            st.subheader("Columns Dtypes")
            for i in self.df:
                st.write(i,"  ", self.df[i].dtype)
            
            st.title("Quick Data Health Check")
            
            data=["Select","Check Null Values","Check Duplicate Values"]
            input=st.selectbox("choose any one",data)
            
            if input==data[1]:
                st.subheader("Null Values")
                
                col1,col2=st.columns(2)
                
                with col1:
                    st.markdown("Before")
                    percent_nan=self.df.isnull().sum()/len(self.df)*100
                    percent_nan=percent_nan[percent_nan > 0]
                    fig1,ax=plt.subplots(figsize=(15,10))
                    sns.barplot(x=percent_nan.index, y=percent_nan.values,palette="viridis",ax=ax)
                    st.pyplot(fig1)
                        
                with col2:
                    
                    st.markdown("After")
                    self.df.dropna(inplace=True)
                    percent_nan=self.df.isnull().sum()/len(self.df)*100
                    fig2,ax=plt.subplots(figsize=(15,10))
                    sns.barplot(x=percent_nan.index, y=percent_nan.values,palette="viridis",ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig2)
                
            elif input==data[2]:
                
                st.subheader("Duplicate Values")
                percent_dup = self.df.duplicated().sum() / len(self.df) * 100
                percent_unique = 100 - percent_dup  # rest are unique

                labels = ["Duplicate Rows", "Unique Rows"]
                sizes = [percent_dup, percent_unique]
                colors = ["#FF6F61", "#726BD6"]

                # Plot pie chart
                fig, ax = plt.subplots(figsize=(6,6))
                ax.pie(sizes, labels=labels, autopct="%.2f%%", startangle=90, colors=colors)
                ax.axis("equal") 
                plt.title("Duplicate vs Unique Rows",pad=25,color="purple")
                st.pyplot(fig)
                
            st.title("Descriptive Analysis")
            st.dataframe(self.df.describe().style.background_gradient(cmap="Blues"))
            
        else:
            st.warning("Please upload a CSV file to proceed.")

class predicter(EDA):
    
    def predict(self):
        
         
        if "df" in st.session_state:
            self.df = st.session_state.df
            st.success("💚 Connected to Weaviate")
            st.title("🎬🍿Movie Magic")
            st.markdown("👋 Welcome to Movie Magic,I'm your AI movie recommender Bot")
            
            movie_list = self.df["Title"].tolist()
            
            selected_movie = st.selectbox("Select a Movie 🌐", movie_list)
            
            st.write(f"You selected: **{(selected_movie)}**")
            
            # st.info("Make Your Prediction By Clicking the Button below:")
            drop_cols = ["Description"]
            self.df = self.df.drop(columns=drop_cols, errors="ignore")

            # # One-hot encode Genre & Certificate
                
            df_encoded = pd.get_dummies(self.df,columns=["Genre"],drop_first=True,dtype=int)

            # # Frequency encode Director & Stars
            col2 = ["Director", "Actors"]
            
            for col in col2:
                if col in df_encoded.columns:
                    freq_map = self.df[col].value_counts().to_dict()
                    df_encoded[col] = df_encoded[col].map(freq_map)
                    
            df_encoded.drop("Title",axis=1,inplace=True)
            df_encoded=df_encoded.dropna()
            valid_idx = df_encoded.dropna().index
            df_encoded = df_encoded.loc[valid_idx].reset_index(drop=True)
            self.df = self.df.loc[valid_idx].reset_index(drop=True)
                        
            operation=make_pipeline(StandardScaler(),KMeans(random_state=101,n_init=10,n_clusters=13))
            operation.fit(df_encoded)
            pre=operation.predict(df_encoded)
            df_encoded["clusters"]=pre
                    # Slider placed before both search modes
            recommed_count = st.slider("Total recommendation", 1, 5, 3)
            setting = st.sidebar.radio("Select your Search Type", ["Normal", "Hybrid"])

            if setting == "Hybrid":
                
                st.sidebar.info("Hybrid Search combines feature-based vectors + clustering + filtering to offer best recommendations")
                
                if st.button("Recommend"):
                    st.write("Here are few recommendations...")

                    # Get cluster of selected movie
                    selected_cluster = df_encoded.loc[self.df["Title"] == selected_movie, "clusters"].values[0]

                    # Get genre(s) of selected movie
                    selected_genre = self.df.loc[self.df["Title"] == selected_movie, "Genre"].values[0]

                    # Filter for same cluster AND same genre
                    mask = (df_encoded["clusters"] == selected_cluster) & (self.df["Genre"] == selected_genre)
                    similar_movies = self.df.loc[mask, "Title"]

                    # Remove the movie itself
                    similar_movies = similar_movies[similar_movies != selected_movie]

                    # Show recommendations
                    if not similar_movies.empty:
                        if len(similar_movies) >= recommed_count:
                            st.success(f"Movies similar to **{selected_movie}** 🎥")
                            st.write(similar_movies.sample(recommed_count).tolist())
                        else:
                            st.warning("try to minimize sample count...")
                    else:
                        st.warning("No similar movies found in this cluster & genre.")

            else:
                st.sidebar.info("Normal Search contain **Genre** filtering to offer recommendations (Disclaimer: Use Hybrid for better recommendations)")

                # Get genre of selected movie
                selected_genre = self.df.loc[self.df["Title"] == selected_movie, "Genre"].values[0]

                # Filter for same genre
                mask = (self.df["Genre"] == selected_genre)
                similar_movies = self.df.loc[mask, "Title"]

                # Remove the movie itself
                similar_movies = similar_movies[similar_movies != selected_movie]

                # Show recommendations
                if not similar_movies.empty:
                    if len(similar_movies) >= recommed_count:
                        st.success(f"Movies similar to **{selected_movie}** 🎥")
                        st.write(similar_movies.sample(recommed_count).tolist())
                    else:
                        st.warning("try to minimize sample count...")
                else:
                    st.warning("No similar movies found in this genre.")



            
        else:
            st.warning("Load **Dataset** first to use this feature....")
                                
            
class stream(predicter):
    
    def run_eda(self):
        self.eda()
        
    def run_prediction(self):
        
        self.predict()
        
    def app(self):
        
        options={"View Data":self.run_eda,
                 "Make Your Predictions":self.run_prediction}
        
        
        key_select=st.sidebar.selectbox("Choose Option",list(options.keys()))
        
        value_select=options[key_select]
        value_select()

str=stream()
str.app()