#for data gathering
import data_vscode 

#for model training
import model_vscode

#for recommendation
import recommend_vscode

import pandas as pd
df=pd.read_csv(r'C:\\Users\\dhima\\infotact\\video_game_data.csv')
data=data_vscode.imp_preprocess(df)
combine,nn,df=model_vscode.model_training(df)

#make the basic frontend structure using streamlit
import streamlit as st
st.title('VIDEO GAME RECOMMENDATION SYSTEM')
st.write('Tired of endlessly scrolling through game libraries, unsure what to play next? GameHub uses intelligent algorithms to recommend video games tailored just for you—based on what you like, what you have played, and what others like you are enjoying')
st.sidebar.title('GAMEHUB MENU')
st.sidebar.title('Search Game Here')
search_game=st.sidebar.text_input('GAME NAME')
search_button=st.sidebar.button('SEARCH')

#for extracting downloaded images in the system
import os
image_folder=r'C:\compressed_v_images'
image_files = {
    os.path.splitext(f)[0].lower(): os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
}

#default structure of 2 columns
column=st.columns(2)
index=0

#for default images on the initial page of streamlit if the user is not searching anything
default_title_df=df[df['rating']<6]
default_title=default_title_df['title'].sample(n=30).str.lower().tolist() #randomly chooses any images of rating upto 5 as default
default_caption='Explore Games'
if not search_game:
    st.subheader(default_caption)
    default_col=st.columns(2)
    default_index=0
    for i in default_title:
        img_path=image_files.get(i)
        if img_path:
            with default_col[default_index%2]:
                st.image(img_path,caption=i,use_container_width=True)  #for default images on the page
            default_index+=1
        else:
            continue


#when the user searches for a game
if search_game and search_button:
    result=recommend_vscode.recommendation(combine,nn,df,search_game,25)   #give the dataframe of recommended games
    try:
        print(result)
        #st.write(result)
        for i in result['title']:
            img_path = image_files.get(i)
            if img_path:
                with column[index%2]:
                    st.image(img_path,caption=i,use_container_width=True)
                index+=1
            else:
                continue

    except:
        st.title('OOPS! NO GAME FOUND')