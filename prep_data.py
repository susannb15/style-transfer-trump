import pandas as pd
from sklearn.model_selection import train_test_split

"""
Prepares and splits the raw tweet data.
"""

trump = pd.read_csv("data/trump.csv")
trump.rename(columns={"Tweet_Text": "text"}, inplace=True)
biden = pd.read_csv("data/biden.csv")
biden.rename(columns={"tweet": "text"}, inplace=True)

pattern = "(www|http:|https:)+[^\s]+[\w]"
filter = trump["text"].str.contains(pattern)
trump = trump[~filter]
filter = biden["text"].str.contains(pattern)
biden = biden[~filter]

trump_tweets = trump["text"].dropna()
biden_tweets = biden["text"].dropna()

trump_tweets.to_csv("data/trump_tweets.csv", header=None, index=None)
biden_tweets.to_csv("data/biden_tweets.csv", header=None, index=None)

trump_tweets = trump_tweets.to_frame(name="text")
trump_tweets["label"] = "1"
biden_tweets = biden_tweets.to_frame(name="text")
biden_tweets["label"] = "0"
trump_tweets["text"].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
biden_tweets["text"].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)

x_bidentrain, x_bidendevtest, y_bidentrain, y_bidendevtest = train_test_split(biden_tweets["text"], biden_tweets["label"], test_size=500, random_state=42) 
x_bidendev, x_bidentest, y_bidendev, y_bidentest = train_test_split(x_bidendevtest, y_bidendevtest, test_size=250, random_state=42) 
biden_traindata = pd.DataFrame(data={"label": y_bidentrain, "text": x_bidentrain})
biden_devdata = pd.DataFrame(data={"label": y_bidendev, "text": x_bidendev})
biden_testdata = pd.DataFrame(data={"label": y_bidentest, "text": x_bidentest})

x_trumptrain, x_trumpdevtest, y_trumptrain, y_trumpdevtest = train_test_split(trump_tweets["text"], trump_tweets["label"], test_size=500, random_state=42)
x_trumpdev, x_trumptest, y_trumpdev, y_trumptest = train_test_split(x_trumpdevtest, y_trumpdevtest, test_size=250, random_state=42)
trump_traindata = pd.DataFrame(data={"label": y_trumptrain, "text": x_trumptrain})
trump_devdata = pd.DataFrame(data={"label": y_trumpdev, "text": x_trumpdev})
trump_testdata = pd.DataFrame(data={"label": y_trumptest, "text": x_trumptest})

data_all = pd.concat([trump_traindata, trump_devdata, biden_traindata, biden_devdata])

#print(data_all.head(5))

x_traindev, x_test, y_traindev, y_test = train_test_split(data_all["text"], data_all["label"], test_size=0.1, random_state=42)
x_train, x_dev, y_train, y_dev = train_test_split(x_traindev, y_traindev, test_size=0.05, random_state=42)

train = pd.DataFrame(data={"text": x_train, "label": y_train})
dev = pd.DataFrame(data={"text": x_dev, "label": y_dev})
test = pd.DataFrame(data={"text": x_test, "label": y_test})

train.to_csv("data/classifier-train.csv", index=None)
dev.to_csv("data/classifier-dev.csv", index=None)
test.to_csv("data/classifier-test.csv", index=None)

biden_traindata.to_csv("data/biden-train.csv", index=None)
biden_devdata.to_csv("data/biden-dev.csv", index=None)
biden_testdata.to_csv("data/biden-test.csv", index=None)

trump_traindata.to_csv("data/trump-train.csv", index=None)
trump_devdata.to_csv("data/trump-dev.csv", index=None)
trump_testdata.to_csv("data/trump-test.csv", index=None)
