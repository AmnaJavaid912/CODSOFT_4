import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns 
import requests
from datasets import load_dataset
from st_aggrid import AgGrid
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

with open('style.css') as f:
    st.markdown( f"<style>{f.read()}</style>" ,unsafe_allow_html=True)

def show_datset_size(dataset):
   st.text(dataset.shape)

@st.cache_data
def load_data():
    url = "ashraq/movielens_ratings"
    try:
        # Loading dataset from hugging face
        
        dataset = load_dataset(url)
        movies_dataset = dataset['train']
        return movies_dataset.to_pandas()
    except requests.exceptions.ConnectionError:
        print("ConnectionError: Couldn't reach the URL on the Hub.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def recommend_me(user_input, rating_pivot_table, similarity_score, filtered_movies): 
    # Recommend movies
    if user_input: 
        recommended_movies_list = [] 
        # Check if the movie title exists in the DataFrame
        is_present = user_input in filtered_movies['title'].values
        if (is_present):
            index = np.where(rating_pivot_table.index == user_input)[0][0]    # find the index of the movie_name
            similar_movies = sorted(list(enumerate(similarity_score[index])), key = lambda x:x[1], reverse = True)[1:6]   # give similarity_Score with index in highest to lower order
            for m in similar_movies:
                recommended_movies_list.append(rating_pivot_table.index[m[0]])
                #print(rating_pivot_table.index[m[0]]) # print recommended movies name
           
            
            movie_info_df = filtered_movies[filtered_movies['title'].isin(recommended_movies_list)].sort_values('rating', ascending = False)
            st.markdown( "<div class= movie-container>", unsafe_allow_html=True)
            if(not movie_info_df.empty):
                unique_movie_names = set() 
                matching_rows = pd.DataFrame(columns=movie_info_df.columns)
                n = 6
                i = 1
                col = st.columns(n)
                for name in recommended_movies_list:
                        matching_rows = movie_info_df[movie_info_df['title'] == name].iloc[0]
                        if not matching_rows.empty and name not in unique_movie_names:
                            col[i].write(f"<div class=movie-tile><img class=movie-poster src={matching_rows['posters']} alt=Movie Image><div class=movie-title>{matching_rows['title']}</div><div class=movie-genre></div></div>", unsafe_allow_html=True) 
                            unique_movie_names.add(name)
                            if i<2:
                                i = i+1
                            else:
                                i = 1 
            st.markdown("</div>", unsafe_allow_html=True)  
        else:
            st.warning(f"'{user_input}' is not present in the movies data.")


def main():
    st.header("Movie Recommendation System") 
    st.sidebar.header("Collaborative based filtering")
    data_loaded = False
    progress_msg = ""
    train_df =  pd.DataFrame([])
    user_input = ""
    df = pd.DataFrame([])
    rating_pivot_table = pd.pivot_table(df) 

    #st.write("Preprocessing data...")
    train_df = load_data() 
    if(not train_df.empty):  
        
        st.markdown(f"<p class=custom-text>Dataset size is {train_df.shape} </p>", unsafe_allow_html=True)
        
        #  - - - - - - - - -  Performing EDA - - - - - - - - -
        # Checking duplicate rows  
        duplicate_rows = train_df[train_df.duplicated()].count()
        # Drop duplicates value if any exist in our dataset
        df = train_df.drop_duplicates()
        # Drop unusable columns
        df = df.drop('imdbId', axis = 1)
        df = df.drop('tmdbId', axis = 1)
        # checking missing values if found drop null values
        null_data = df.isnull().sum()
        if(null_data.count() > 0):
            df = df.dropna()      

            #  - - - - - - - - -  Collaborative based recommendation system  - - - - - - - - -  

            # Filtering dataframe...
            if(df.empty == False):
                # Users and number of rating they did, in descending order
                users_rating = df.groupby('user_id').count()['rating'].reset_index().sort_values('rating', ascending = False)
                
                #   - - - - - - - - -  Filtering Users  - - - - - - - - -  
                # Selecting users that rated minimum 200 movies
                movie_viewers = users_rating[users_rating['rating'] >= 200]
                filtered_users_df = df[df['user_id'].isin(movie_viewers['user_id'])]
                

                #   - - - - - - - - -  Filtering movies - - - - - - - - -  

                # Selecting movies that have number of ratings above 100
                movies = filtered_users_df.groupby('title').count()['rating'] >= 100
                filtered_movies =  filtered_users_df[filtered_users_df['title'].isin(movies.index)].sort_values('user_id', ascending = False)
                
                # drop duplicate record if any found
                filtered_movies = filtered_movies.drop_duplicates()
                #st.write("Display 10 records from the dataset:")
                AgGrid(filtered_movies.head(6))
                #st.dataframe(filtered_movies.head(6), width = 700, height=250)

                #   - - - - - - - - -  Creating pivot table - - - - - - - - -  
                #st.write("Creating pivot table...")
                rating_pivot_table = filtered_movies.pivot_table(index='title', columns='user_id', values = 'rating')
                rating_pivot_table.fillna(0, inplace = True)
                
                #  - - - - - - - - -  Calculating Euclidean Distance - - - - - - - - -  
                #st.write("Finding similarity score for each movie...")

                # Finding similarity score of each movie
                similarity_score = cosine_similarity(rating_pivot_table)
                
                # finding correlation of features
                #df.corr()
                st.success("Data loaded successfully!")
                data_loaded = True
    elif(train_df.empty == True):
        st.write("No data loaded.")  
    if data_loaded:
        user_input = st.sidebar.text_input("Enter the movie name you recently watched:") 
        st.sidebar.button("Recommend", on_click=recommend_me(user_input, rating_pivot_table, similarity_score, filtered_movies))
    
    
                
        
 
if __name__ == "__main__":
    main()
