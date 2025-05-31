import pandas as pd

def recommendation(combine,nn,df,game_title,top_games=5):
    try:
        game_index=df[df['title']==game_title.lower()].index[0]

        # NearestNeighbors model to find closest games based on combined features
        dist,indices=nn.kneighbors(combine[game_index],n_neighbors=top_games+1)

        #original game that searched is also included in recommendation
        searched=df[df['title']==game_title][['title', 'positive_ratio', 'user_reviews', 'tags']]

        #other games that model recommends
        recommend=df.iloc[indices[0][1:]][['title','positive_ratio','user_reviews','tags']]

        #concatenate together the original game and the recommended games
        result=pd.concat([searched,recommend])

        #return the dataframe of recommended game and the original game
        return result
    
    #if the searched game is not in the database
    except:
        return 'NO DATA FOUND'

game_title='grand theft auto'
#result=recommendation(combine,nn,df,game_title,5)