import luigi
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.
        Output file will contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='Data/Processed/clean_data.csv')
          
    def run(self):
        # Importing the file
        df_tweet = pd.read_csv(self.tweet_file, engine='python')
        
        # Dropping all columns except airline_sentiment and tweet_coord
        df_tweet.drop(df_tweet.columns.difference(['airline_sentiment','tweet_coord']), axis=1, inplace=True)
        # Drop all Null values
        df_tweet.dropna(inplace=True)
        # Split the Lat and Long
        df_tweet[['lat', 'long']] = df_tweet['tweet_coord'].str.split(',',expand=True)
        # Clean the values
        df_tweet['lat'] = df_tweet['lat'].str.replace('[', '')
        df_tweet['long'] = df_tweet['long'].str.replace(']', '')
        df_tweet.drop(['tweet_coord'],axis=1,inplace=True)
        # Drop the zero coordinates
        df_tweet = df_tweet[~(df_tweet['lat'].isin(['0.0'])) & ~(df_tweet['long'].isin(['0.0']))]
        
        # Writing to the output
        df_tweet.to_csv(self.output().path, index=False)
        
    def output(self):
        return luigi.LocalTarget(self.output_file)
        
        
class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.
        Output file will have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='Data/Raw/cities.csv')
    output_file = luigi.Parameter(default='Data/Processed/features.csv')
    
    def requires(self):
        return CleanDataTask(tweet_file=self.tweet_file)
        
    def run(self):
        # Importing the file
        df_tweet = pd.read_csv(self.input().path, engine='python')
        cities = pd.read_csv(self.cities_file, engine='python')
        # Using Label encoder to convert airline_sentiment to numerical values
        encoder = LabelEncoder()
        df_tweet['airline_sentiment'] = encoder.fit_transform(df_tweet['airline_sentiment'])
        
        # Using KNN to find the nearest city by calculating Euclidean distance
        cities_coor = cities[['latitude', 'longitude']]
        cities_name = cities.asciiname
        
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(cities_coor, cities_name)
        
        tweet_coor = df_tweet[['lat', 'long']]
        df_tweet['city'] = knn.predict(tweet_coor)
        
        # Arranging the variables and one hot encoding the city
        df_tweet = df_tweet[['city', 'airline_sentiment']]
        df_city = pd.get_dummies(df_tweet['city'])
        df_tweet = pd.concat([df_city, df_tweet], axis=1)

        # Writing to the file
        df_tweet.to_csv(self.output().path, index=False)
        
    def output(self):
        return luigi.LocalTarget(self.output_file)


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.
        Output file will be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='Data/Results/model.pkl')
    
    def requires(self):
        return TrainingDataTask(tweet_file=self.tweet_file)

    def run(self):
        # Importing the file
        df_tweet = pd.read_csv(self.input().path, engine='python')
        
        # Spliting target variable and cities
        X_cities = df_tweet.drop(['airline_sentiment', 'city'], axis=1)
        y_sen = df_tweet.airline_sentiment
        
        # Training RandomForestClassifier using the available data
        forest = RandomForestClassifier()
        forest.fit(X_cities, y_sen)
        
        # Dumping the fitted model
        with open(self.output_file, 'wb') as model:
            pickle.dump(forest, model)
 
    def output(self):
        return luigi.LocalTarget(self.output_file) 


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.
        Output file will be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    cleaned_tweetfile = luigi.Parameter(default='Data/Processed/features.csv')
    output_file = luigi.Parameter(default='Data/Results/scores.csv')

    def requires(self):
        return TrainModelTask(tweet_file=self.tweet_file)
    
    def run(self):
        # Importing the file
        df_tweet = pd.read_csv(self.cleaned_tweetfile, engine='python')
        
        # Separating the city
        X_cities = df_tweet.drop(['airline_sentiment', 'city'], axis=1)
        
        # Loading the model from pickle file
        with open(self.input().path, 'rb') as model:
            forest = pickle.load(model)
            
        # Predicting the probability of each sentiment based on city
        prediction = pd.DataFrame(forest.predict_proba(X_cities), columns=['Negative', 'Neutral', 'Positive'])
        df_tweet = pd.concat([df_tweet, prediction], axis=1)
        output_score = df_tweet[['city', 'Negative', 'Neutral', 'Positive']]
        # Since there are many rows for each city, lets take only one row per city
        output_score.drop_duplicates(inplace=True)
        output_score.sort_values(by=['Positive'], ascending=False, inplace=True)
        
        output_score.to_csv(self.output().path, index=False)

    def output(self):
        return luigi.LocalTarget(self.output_file) 
        
        
if __name__ == "__main__":
    luigi.run()
