import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from IPython.display import FileLink

# Fonction pour charger les fichiers de profils
def load_user_data(file_path, label):
    df = pd.read_csv(file_path, sep="\t", header=None)
    df.columns = ["UserID", "CreatedAt", "CollectedAt", "NumberOfFollowings", "NumberOfFollowers", 
                      "NumberOfTweets", "LengthOfScreenName", "LengthOfDescriptionInUserProfile"]
    df['CreatedAt'] = pd.to_datetime(df['CreatedAt'])
    df['CollectedAt'] = pd.to_datetime(df['CollectedAt'])
    df['Account_Lifespan'] = (df['CollectedAt'] - df['CreatedAt']).dt.days
    df['Following_Followers_Ratio'] = df['NumberOfFollowings'] / (df['NumberOfFollowers'] + 1)
    df['Avg_Tweets_Per_Day'] = df['NumberOfTweets'] / (df['Account_Lifespan'] + 1)
    df['Label'] = label  # 1 = pollueur, 0 = légitime
    return df
    
# Charger les fichiers de profils
polluters = load_user_data("data/content_polluters.txt", 1)
legitimate = load_user_data("data/legitimate_users.txt", 0)

users_data = pd.concat([polluters, legitimate], ignore_index=True)

# Check first rows
print(users_data.head())


# Fonction pour charger les fichiers de tweets
def load_tweet_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["UserID", "TweetID", "Tweet", "CreatedAt"])
    df['CreatedAt'] = pd.to_datetime(df['CreatedAt'])
    return df

# Charger les fichiers de tweets
polluters_tweets = load_tweet_data("data/content_polluters_tweets.txt")
legitimate_tweets = load_tweet_data("data/legitimate_users_tweets.txt")

# Fonction pour calculer les caractéristiques des tweets
def compute_tweet_features(tweets_df):
    # Trier les tweets par UserID et CreatedAt pour éviter les valeurs négatives dans diff()
    tweets_df = tweets_df.sort_values(by=['UserID', 'CreatedAt'])

    # Calcul des différentes métriques par utilisateur
    tweet_features = tweets_df.groupby("UserID").agg(
        URL_Proportion=('Tweet', lambda x: np.mean(x.str.contains('http', na=False))),
        Mention_Proportion=('Tweet', lambda x: np.mean(x.str.contains('@', na=False))),
        Hashtag_Proportion=('Tweet', lambda x: np.mean(x.str.contains("#", na=False))),
        Avg_Time_Between_Tweets=('CreatedAt', lambda x: x.diff().mean().total_seconds() if len(x) > 1 else 0),
        Max_Time_Between_Tweets=('CreatedAt', lambda x: x.diff().max().total_seconds() if len(x) > 1 else 0)
    ).reset_index()

    return tweet_features







# Calculer les caractéristiques des tweets
polluters_tweet_features = compute_tweet_features(polluters_tweets)
legitimate_tweet_features = compute_tweet_features(legitimate_tweets)

# Fusionner les datasets de profils et de tweets
final_polluters = polluters.merge(polluters_tweet_features, on='UserID', how='left')
final_legitimate = legitimate.merge(legitimate_tweet_features, on='UserID', how='left')

# Fonction pour charger les fichiers followings et calculer le ratio
def load_following_variability(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["UserID", "Followings"])
    df["Followings"] = df["Followings"].apply(lambda x: list(map(int, x.split(","))))
    df["Max_Followings"] = df["Followings"].apply(max)
    df["Min_Followings"] = df["Followings"].apply(min)
    df["Following_Variability_Ratio"] = df["Max_Followings"] / (df["Min_Followings"] + 1)
    return df[["UserID", "Following_Variability_Ratio"]]

# Charger les fichiers followings
polluters_followings = load_following_variability("data/content_polluters_followings.txt")
legitimate_followings = load_following_variability("data/legitimate_users_followings.txt")

# Fusionner les ratios de variabilité des followings avec les datasets
final_polluters = final_polluters.merge(polluters_followings, on='UserID', how='left')
final_legitimate = final_legitimate.merge(legitimate_followings, on='UserID', how='left')






# Fusionner tous les utilisateurs en un seul dataset
data = pd.concat([final_polluters, final_legitimate], ignore_index=True)

# Réorganisation des cols
column_order = [
    "LengthOfScreenName", "LengthOfDescriptionInUserProfile", "Account_Lifespan", "NumberOfFollowings", 
    "NumberOfFollowers", "Following_Followers_Ratio", "Avg_Tweets_Per_Day", "URL_Proportion", 
    "Mention_Proportion", "Hashtag_Proportion", "Avg_Time_Between_Tweets", "Max_Time_Between_Tweets", 
    "Following_Variability_Ratio", "Label"
]
data = data[column_order]
print(data.tail(100))

# Nettoyage des données
data = data.drop_duplicates()
data.fillna(data.median(), inplace=True)

# Normalisation (Z-score)
scaler = StandardScaler()
feature_cols = [
    'LengthOfScreenName', 'LengthOfDescriptionInUserProfile', 'Account_Lifespan',
    'NumberOfFollowings', 'NumberOfFollowers', 'Following_Followers_Ratio',
    'Avg_Tweets_Per_Day', 'URL_Proportion', 'Mention_Proportion', 'Hashtag_Proportion',
    'Avg_Time_Between_Tweets', 'Max_Time_Between_Tweets', 'Following_Variability_Ratio'
]





data[feature_cols] = scaler.fit_transform(data[feature_cols])

# Ajouter les 2 nouvelles caractéristiques
data["Hashtag_Ratio"] = data["Hashtag_Proportion"] / (data["Avg_Tweets_Per_Day"] + 1)
data["Time_Between_Variability"] = data["Max_Time_Between_Tweets"] - data["Avg_Time_Between_Tweets"]

# Mettre "Label" à la fin pour garder la cohérence
cols = [col for col in data.columns if col != "Label"] + ["Label"]
data = data[cols]  # Réorganiser automatiquement

# Vérification des nouvelles colonnes
print("\nAperçu des nouvelles caractéristiques :")
print(data[["Hashtag_Ratio", "Time_Between_Variability"]].head())

# Sauvegarde du dataset mis à jour
data.to_csv("final_dataset_updated.csv", index=False)
print(" Le dataset mis à jour a été sauvegardé sous 'final_dataset_updated.csv'")
FileLink("final_dataset_updated.csv")


