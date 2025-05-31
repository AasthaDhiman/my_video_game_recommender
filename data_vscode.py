import pandas as pd
import numpy as np
import matplotlib as plt
import ast

df=pd.read_csv(r'C:\\Users\\dhima\\infotact\\video_game_data.csv')

#for general preprocessing of data 
def imp_preprocess(dataset):
    dataset['tags']=dataset['tags'].apply(lambda x:" ".join(ast.literal_eval(x))) #change the datatype of tags to str
    
    #encoded to numerical values
    dataset['win']=dataset['win'].map({True:1,False:0}) 
    dataset['mac']=dataset['mac'].map({True:1,False:0})
    dataset['linux']=dataset['linux'].map({True:1,False:0})
    
    dataset['rating']=dataset['rating'].map({'Overwhelmingly Positive':1,'Very Positive':2,'Positive':3,'Mostly Positive':4,'Mixed':5,'Mostly Negative':6,'Negative':7,'Very Negative':8,'Overwhelmingly Negative':9})
    
    #normalize the data so that the range of user_reviews becomes less as it is right skewed
    dataset['user_reviews'] = np.log1p(dataset['user_reviews'])  # handles 0 safely
    
    #change title to lowercase
    dataset['title']=dataset['title'].str.lower()
    dataset=dataset[~dataset['title'].duplicated(keep=False)]
    return dataset
#df=imp_preprocess(df)

#print(df.sample(5))
