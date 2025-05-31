from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
rb=RobustScaler()
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.neighbors import NearestNeighbors

def model_training(df):
    #features that need scaling
    to_scale=['positive_ratio','user_reviews']
    X=df[to_scale]

    #split data into train-test-split
    xtrain,xtest=train_test_split(X,test_size=0.2,random_state=42)

    #robust scaler on train data
    xtrain_scaled=rb.fit(xtrain)

    #scale all values-train and test both
    X_scaled=rb.transform(X)

    #convert the tags to numerical vectors
    #stopwords(word used in sentence making but has no meaning) removed
    tfidf=TfidfVectorizer(stop_words='english')
    tfidf_tags=tfidf.fit_transform(df['tags'].fillna(''))

    #horizontally combine the sparse matrix and features
    combine=hstack([X_scaled,tfidf_tags])

    #compressed for efficiency -it takes extreme memory if not compressed
    combine=combine.tocsr()

    #applying KNN using cosine similarity and brute force approach
    nn=NearestNeighbors(metric='cosine',algorithm='brute')
    nn.fit(combine)

    #returns following
    return combine,nn,df

#combine,nn,df=model_training(df)
