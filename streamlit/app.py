import streamlit as st
import pandas as pd
import numpy as np
#import pickel
st.set_page_config(page_title="Cricket player prediction",page_icon=":tada",layout="wide")
#loaded_model=pickle.load("C:/Users\SGOUD\OneDrive - Capgemini\Desktop\hackathon\data\trained_model.sav"
st.subheader("Hi, there :wave: , hope you are doing good!")
st.title("Cricket player prediction")
st.write("Created a recommendation engine to predict the top 10 players ")

df=pd.read_excel("C:/Users/SGOUD/OneDrive - Capgemini/Desktop/hackathon/data/output.xlsx",sheet_name=0)

df = df.drop(['Unnamed: 0'], axis=1)


df["Status"]="Active"

import glob

df_batting = pd.concat(map(pd.read_excel, glob.glob('C:/Users/SGOUD/OneDrive - Capgemini/Desktop/hackathon/data/battingdata/*.xlsx')))
df_bowling = pd.concat(map(pd.read_excel, glob.glob('C:/Users/SGOUD/OneDrive - Capgemini/Desktop/hackathon/data/bowlingdata/*.xlsx')))


df_merged=df_batting.merge(df_bowling, how='outer', on=['Player','Oppositeteam'])

df_m=df_merged.rename(columns = lambda x: x.replace('_x', '_batting') if x !=50 and x !=100 else x)
df_m=df_m.rename(columns = lambda x: x.replace('_y', '_bowling') if x !=50 and x !=100 else x)

df_m['BBI_FF'] = df_m['BBI_FF'].apply(lambda x: eval(str(x)) if '/' in str(x) else float(x))

df_m = df_m.replace(np.nan, 0)


df_m = df_m.drop(['BBI','Batting_batting','Bowling_batting','Allrounder_batting','Batting_bowling','Bowling_bowling','Allrounder_bowling'], axis=1)


# In[15]:



# In[16]:


df_m["Style"] = df_m["Player"].map(df.set_index('Player')["Style"])


# In[17]:



# In[18]:


df_m["Status"] = df_m["Player"].map(df.set_index('Player')["Status"])


# In[19]:



# In[20]:


df_s=df_m.loc[df_m["Status"]=="Active"]# and df_s["Status"]=="Inactive"].any()


# In[21]:


df_s["Player"].nunique()


# In[22]:




# In[23]:


df_f=df_s.copy()
df_f= df_f.reset_index()


# In[24]:






df_f = df_f.drop(['index'], axis=1)



df_f = df_f.drop(['Status'], axis=1)

# In[28]:



# In[30]:


c = ['Player', 'I_batting', 'R_batting', 'B',
     'Outs', 'Avg_batting', 'SR_batting', 'HS',
     '4s_batting', '6s_batting', 50, 100,
     'Oppositeteam', 'I_bowling', 'O', 'R_bowling',
     'W', 'Econ', 'Avg_bowling', 'SR_bowling',
     '4W', '5W', '4s_bowling', '6s_bowling',
     'Dots', 'BBI_FF', 'Style']

# In[31]:


df_f["discription"] = df_f[c].astype(str).apply(' '.join, axis=1)

# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# In[33]:


# Let's assume we have some data in 'dataframe' and 'player_stats' is the feature on which we want to base our recommendations
tfidf = TfidfVectorizer(stop_words='english')
# dataframe['player_stats'] = dataframe['player_stats'].fillna('')
tfidf_matrix = tfidf.fit_transform(df_f["discription"])
# df["description"]=

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)



def get_recommendations(df, opposite_team, cosine_sim=cosine_sim):
    player_index = df_f[(df_f["Oppositeteam"] == opposite_team)].index.tolist()
    sim_scores_l = []
    for team_index in player_index:
        sim_scores = list(enumerate(cosine_sim[team_index]))
        # Sort the players based on the cosine similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores_l.extend(sim_scores)
        # sim_scores_l=sorted(set(sim_scores_l), key=lambda x: x[1], reverse=True)[:10]
    seen_indices = set()
    unique_sim_scores = []
    for player_index, score in sorted(sim_scores_l, key=lambda x: x[1], reverse=True):
        if player_index not in seen_indices:
            seen_indices.add(player_index)
            unique_sim_scores.append((player_index, score))
    unique_top_10 = unique_sim_scores[:10]

    # Get the player indices
    player_indices = [i[0] for i in unique_top_10]
    # df['Style'] = np.where(df['Style'] == 1, "Batsman", np.where(df['Style'] == 0, "Bowler","Allrounder"))

    # return df['Player'].iloc[player_indices].tolist()
    return df[['Player', "Style"]].iloc[player_indices].reset_index(drop=True)


# In[35]:


u_p = df_f["Player"].unique()
u_t = df_f["Oppositeteam"].unique()

# In[36]:

op=st.text_input("Enter the any country name among   Australia, Bangladesh, England, New Zealand, Pakistan,South Africa, Sri Lanka")
# op = input(
#     'enter the any country name among  Australia, Bangladesh, England, New Zealand, Pakistan,South Africa, Sri Lanka')
r = get_recommendations(df_f, op)
r['Style'] = np.where(r['Style'] == 1, "Batsman", np.where(r['Style'] == 0, "Bowler", "Allrounder"))
# print(f"top 10 recomended players against {op}:")
# print(r)
st.dataframe(r)
