##################################################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
##################################################

# Review: Comment of product
# Title: The title given to the content of the comment, a short comment
# HelpFul: The number of people who found the comment useful
# Star: The number of stars awarded to the product

# Tasks
##############################################################

# Task 1: Text preprocessing operations.
        # 1. amazon.read the xlsx data.
        # 2. On the "review" variable
            # a. Convert all letters to lowercase
            # b. Remove the punctuation marks
            # c. Remove the numerical expressions in the comments
            # D. Remove words (stopwords) that do not contain information from the data
            # e. remove less than 1000 words from the data
            # f. Apply the lemmatization process

# Task 2: Text Visualization
        # Step 1: Barplot visualization process
                  # a. Calculate the frequencies of the words contained in the "Review" variable, save them as tf
                  # b. rename the columns of the tf dataframe as: "words", "tf"
                  # c. Complete the visualization process with barplot by filtering the value of the variable "tf" according to those that are more than 500.

       # Step 2: The WordCloud visualization process
                 # a. Save all the words contained in the "review" variable as a string in the name "text"
                 # b. Define and save your template shape using WordCloud
                 # c. Generate the wordcloud you saved with the string you created in the first step.
                 # D. Complete the visualization steps. (figure, imshow, axis, show)

# Task 3: Emotion Analysis
      # Step 1: Create the SentimentIntensityAnalyzer object defined in the NLTK package in Python

      # Step 2: Investigation of polarity scores with sentimentintensitiyanalyzer object
                # a. Calculate polarity_scores() for the first 10 observations of the "Review" variable
                # b. For the first 10 observations examined, please observe again by filtering according to the compund scores
                # c. if the compound scores for 10 observations are greater than 0, update them as "neg" if not "pos"
                # D. Add it to the dataframe as a new variable by assigning pos-neg for all observations in the "Review" variable


# Task 4: Preparing for machine learning!
        # Step 1: Divide the data into train tests by determining our dependent and independent variables.
        # Step 2: We need to convert the representation shapes to digital in order to provide the data to the machine learning model.
                  # a. Create an object using the TfidfVectorizer.
                  # b. Please fit the object we have created using our train data that we have previously allocated.
                  # c. Apply the transformation process and save the vector we have created to the train and test data.

# Task 5: Modeling (Logistic Regression)
    # Step 1: Fit with the train data by setting up the logistic regression model.
    # Step 2: Perform prediction operations with the model you have installed.
        # a. Record the test data by estimating it with the Predict function.
        # b. report and observe your forecast results with classification_report.
        # c. calculate the average accuracy value using the cross validation function
   # Step 3: Asking the model by selecting the rating from the comments contained in the data.
        # a. select a sample from the "Review" variable with the sample function and assign it to a new value
        # b. Vectorize the sample you have obtained with the CountVectorizer so that the model can predict.
        # c. Record the sample you have vectorized by performing the fit and transform operations.
        # D. Record the prediction result by giving the sample to the model you have set up.
        # e. Print the sample and the forecast result on the screen.

# Task 6: Modeling (Random Forest)
        # Step 1: Viewing the results of the Random Forest model;
                 # a. Install and fit the RandomForestClassifier model.
                 # b. calculate the average accuracy value using the cross validation function
                 # c. Compare the results with the logistic regression model.


############################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)


# TEXT PRE-PROCESSING

df = pd.read_excel("amazon.xlsx")
df.head()
df.info()

# Normalizing Case Folding
df['Review'] = df['Review'].str.lower()

# Punctuations
df['Review'] = df['Review'].str.replace('[^\w\s]', '')

# Numbers
df['Review'] = df['Review'].str.replace('\d', '')

# Stopwords
# nltk.download('stopwords')
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# Rarewords / Custom Words
sil = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

# Lemmatization
# nltk.download('wordnet')
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df['Review'].head(10)

# Barplot
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

# Wordcloud
text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

sia = SentimentIntensityAnalyzer()


df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df.groupby("Sentiment_Label")["Star"].mean()


# Test-Train
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)

# TF-IDF Word Level
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

y_pred = log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, test_y))

cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()

random_review = pd.Series(df["Review"].sample(1).values)
yeni_yorum = CountVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(yeni_yorum)
print(f'Review:  {random_review[0]} \n Prediction: {pred}')

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()

