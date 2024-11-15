import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load clustered song data and map descriptive names to each cluster
df_songs = pd.read_csv('/Users/katyakraft/Documents/IronHack/project_week_10/clustered_songs.csv')

# Define cluster names
cluster_names = {
    0: "Laid-Back Beats",
    1: "Groove Warm-Up",
    2: "Dance Floor Ready",
    3: "All-Out Party"
}
df_songs['cluster_name'] = df_songs['cluster'].map(cluster_names)
df_songs = df_songs[df_songs['popularity'] > 20].reset_index(drop=True)

# Streamlit app layout
st.title("Party-Ready Song Recommender")
st.write("Enter a song, and we’ll find some party-ready tracks for you to complete your playlist!")
st.image("icon2.jpg", width=550)

# Display the head of the DataFrame as an overview
st.write("Overview of the dataset:")
st.dataframe(df_songs[['name', 'artist', 'cluster_name', 'cluster', 'popularity']].head())

# User input for song recommendation
st.subheader("Find Party-Ready Song Recommendations")
user_song = st.text_input("Enter a song name:")

# Process user input to find similar songs
if user_song:
    # Search for the song in the DataFrame
    user_cluster = df_songs[df_songs['name'].str.contains(user_song, case=False)]
    
    if not user_cluster.empty:
        # Extract features of the matched song
        song_features = user_cluster[['danceability', 'energy']].values[0].reshape(1, -1)
        cluster_id = user_cluster['cluster'].iloc[0]
        
        # Filter DataFrame to the same cluster as the selected song
        party_songs = df_songs[(df_songs['cluster'] == cluster_id)]
        
        # Calculate similarity based on danceability and energy
        similarities = cosine_similarity(song_features, party_songs[['danceability', 'energy']])
        party_songs['similarity'] = similarities[0]
        
        # Sort by similarity for top recommendations
        recommendations = party_songs.sort_values(by="similarity", ascending=False).head(10)
        
        # Display recommendations in a table format with specified columns
        st.subheader("Recommended Party-Ready Songs")
        st.table(recommendations[['name', 'artist', 'popularity', 'cluster_name', 'cluster']])
    else:
        st.write("Song not found in the database. Try a different song.")