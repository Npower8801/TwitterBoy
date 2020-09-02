from flask import Flask, render_template, request
import pickle
import numpy as np
import tweepy

app = Flask(__name__)

model = pickle.load(open("model.pkl", 'rb'))
vectorizer_obj = pickle.load(open("vectorizer.pkl", 'rb'))

API_KEY = "3L094jJzc7ReFcSJRaDEIbxMA"
API_SECRET_KEY = "gMGywEFtXWBbz9SDjjLNOc4PeWCe7zEKvREgqDOqNEP6uk0ycq"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAFwnGwEAAAAANSEpy5uavqoVgdnxhuYZKJEi4N8%3DiOAenc38m1jO4MfTIjQMGAvfih5T6LP18tyQ496y15VORsTATJ"
ACCESS_TOKEN = "1246008382323998721-OH4KwSWVTYduzE3OvxZL4eb10QzIsz"
ACCESS_TOKEN_SECRET = "dXITbcdlUy5sh8yYLmTZkSDTyTt4mINkROb83CI6GKVK7"


def get_tweets(username):
    # Authorization to consumer key and consumer secret
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)

    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    api = tweepy.API(auth)

    tweets = api.user_timeline(screen_name=username, count=100,tweet_mode="extended")

    tmp = []

    tweets_for_csv = [tweet.full_text for tweet in tweets if tweet.lang == "en"]  # CSV file created
    for j in tweets_for_csv:
        # Appending tweets to the empty array tmp
        tmp.append(j)
    return np.array(tmp)

def Clean_tweets(tweets_list):

    import nltk,re
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    cleaned_tweets_list = []
    for i in range(0, len(tweets_list)):
        tweets_cleaned = re.sub('http[s]?://\S+', '', tweets_list[i])
        tweets_cleaned = re.sub('[^a-zA-Z]', ' ',tweets_cleaned)
        tweets_cleaned = tweets_cleaned.lower()
        tweets_cleaned = tweets_cleaned.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        tweets_cleaned = [ps.stem(word) for word in tweets_cleaned if not word in set(all_stopwords)]
        tweets_cleaned = ' '.join(tweets_cleaned)
        cleaned_tweets_list.append(tweets_cleaned)

    return cleaned_tweets_list


@app.route('/')
def twitterboy():
    return render_template("index.html")


@app.route('/result_analysis', methods=['POST', 'GET'])
def result_analysis():
    if request.method == 'POST':
        twitter_handle = request.form['twitter_handle_typed']
        if twitter_handle == " ":
            return False
        else:
            top_tweets = get_tweets(twitter_handle)
            print(top_tweets[0])
            top_tweets_cleaned = Clean_tweets(top_tweets)
            vectors_words = vectorizer_obj.transform(top_tweets_cleaned).toarray()
            predection = model.predict(vectors_words)
            main_prediction = []
            postive_counter = 0
            negative_counter = 0

            for _ in predection:
                if _==0:
                    main_prediction.append("Negative")
                    negative_counter = negative_counter + 1
                else:
                    main_prediction.append("Positive")
                    postive_counter = postive_counter+1
                result = zip(top_tweets,main_prediction)

            return render_template("index.html" , output_result = result,handle="Recent Tweets from twitter handle : @"+twitter_handle,counter_positive="Total no. of positive tweets : "+str(postive_counter),counter_negative="Total no. of negative tweets : "+str(negative_counter))


if __name__ == '__main__':
    app.run()
