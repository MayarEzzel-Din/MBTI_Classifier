import pandas as pd
import numpy as np
import re

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import warnings
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")


'''read data'''
data = pd.read_csv("mbti_1.csv")
#print(data.head(10))

'''List of posts'''
[p.split('|||') for p in data.head(2).posts.values]


'''Distribution of the MBTI personality types'''
cnt_types = data['type'].value_counts()

plt.figure(figsize=(12,4))
sns.barplot(cnt_types.index, cnt_types.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Types', fontsize=12)
plt.show()

'''Add columns for the type Indicators'''
def get_types(row):
    t = row['type']

    I = 0;
    N = 0
    T = 0;
    J = 0

    if t[0] == 'I':
        I = 1
    elif t[0] == 'E':
        I = 0
    else:
        print('I-E incorrect')

    if t[1] == 'N':
        N = 1
    elif t[1] == 'S':
        N = 0
    else:
        print('N-S incorrect')

    if t[2] == 'T':
        T = 1
    elif t[2] == 'F':
        T = 0
    else:
        print('T-F incorrect')

    if t[3] == 'J':
        J = 1
    elif t[3] == 'P':
        J = 0
    else:
        print('J-P incorrect')
    return pd.Series({'IE': I, 'NS': N, 'TF': T, 'JP': J})


data = data.join(data.apply(lambda row: get_types(row), axis=1))
#print(data.head(5))

#print ("Introversion (I) /  Extroversion (E):\t", data['IE'].value_counts()[0], " / ", data['IE'].value_counts()[1])
#print ("Intuition (N) – Sensing (S):\t\t", data['NS'].value_counts()[0], " / ", data['NS'].value_counts()[1])
#print ("Thinking (T) – Feeling (F):\t\t", data['TF'].value_counts()[0], " / ", data['TF'].value_counts()[1])
#print ("Judging (J) – Perceiving (P):\t\t", data['JP'].value_counts()[0], " / ", data['JP'].value_counts()[1])

N = 4
but = (data['IE'].value_counts()[0], data['NS'].value_counts()[0], data['TF'].value_counts()[0], data['JP'].value_counts()[0])
top = (data['IE'].value_counts()[1], data['NS'].value_counts()[1], data['TF'].value_counts()[1], data['JP'].value_counts()[1])

ind = np.arange(N)    # the x locations for the groups
width = 0.7      # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, but, width)
p2 = plt.bar(ind, top, width, bottom=but)

plt.ylabel('Count')
plt.title('Distribution accoss types indicators')
plt.xticks(ind, ('I/E',  'N/S', 'T/F', 'J/P',))

plt.show()

'''Pearson Features Correlation'''
data[['IE', 'NS', 'TF', 'JP']].corr()

'''Correlation'''
cmap = plt.cm.RdBu
corr = data[['IE','NS','TF','JP']].corr()
plt.figure(figsize=(12,10))
plt.title('Pearson Features Correlation', size=15)
sns.heatmap(corr, cmap=cmap,  annot=True, linewidths=1)
plt.show()

'''Prep data'''
b_Pers = {'I': 0, 'E': 1, 'N': 0, 'S': 1, 'F': 0, 'T': 1, 'J': 0, 'P': 1}
b_Pers_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]


def translate_personality(personality):
    # transform mbti to binary vector

    return [b_Pers[l] for l in personality]


def translate_back(personality):
    # transform binary vector to mbti personality

    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s


# Check ...
d = data.head(4)
list_personality_bin = np.array([translate_personality(p) for p in d.type])
#print("Binarize MBTI list: \n%s" % list_personality_bin)

##### Compute list of subject with Type | list of comments


'''We want to remove these from the posts'''
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                    'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

unique_type_list = [x.lower() for x in unique_type_list]

# Lemmatize
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

# Cache the stop words for speed
cachedStopWords = stopwords.words("english")


def pre_process_data(data, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []
    len_data = len(data)
    i = 0

    for row in data.iterrows():
        i += 1
        #if (i % 500 == 0 or i == 1 or i == len_data):
         #   print("%s of %s rows" % (i, len_data))

        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality

list_posts, list_personality  = pre_process_data(data, remove_stop_words=True)

#print("Num posts and personalities: ",  list_posts.shape, list_personality.shape)

'''Vectorize with count and tf-idf'''
# Posts to a matrix of token counts
cntizer = CountVectorizer(analyzer="word",
                             max_features=1500,
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_df=0.7,
                             min_df=0.1)

# Learn the vocabulary dictionary and return term-document matrix
#print("CountVectorizer...")
X_cnt = cntizer.fit_transform(list_posts)

# Transform the count matrix to a normalized tf or tf-idf representation
tfizer = TfidfTransformer()

#print("Tf-idf...")
# Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
X_tfidf =  tfizer.fit_transform(X_cnt).toarray()

feature_names = list(enumerate(cntizer.get_feature_names()))
#print(feature_names)
#print(X_tfidf.shape)

'''Train XGBoost classifiers'''
#print("X: Posts in tf-idf representation \n* 1st row:\n%s" % X_tfidf[0])

type_indicators = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) – Sensing (S)",
                   "FT: Feeling (F) - Thinking (T)", "JP: Judging (J) – Perceiving (P)"  ]

#for l in range(len(type_indicators)):
 #   print(type_indicators[l])

#print("MBTI 1st row: %s" % translate_back(list_personality[0,:]))
#print("Y: Binarized MBTI 1st row: %s" % list_personality[0,:])

# Posts in tf-idf representation

X = X_tfidf

###################################################XGBoost
'''Training & Testing XGBoost'''
'''
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    # Let's train type indicator individually
    Y = list_personality[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s XGBoost Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))
'''

##################################################SVM Training

for l in range(len(type_indicators)):
    #print("%s ..." % (type_indicators[l]))

    # Let's train type indicator individually
    Y = list_personality[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    SVM_model = OneVsRestClassifier(SVC(kernel='linear', C=0.1)).fit(X_train, y_train)

    # make predictions for test data
    y_pred = SVM_model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s SVM Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))


##################################################KNN Training
'''
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    # Let's train type indicator individually
    Y = list_personality[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    KNN_model = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train)

    # make predictions for test data
    y_pred = KNN_model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s KNN Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))
'''

##################################################Naive Bayes
'''
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    # Let's train type indicator individually
    Y = list_personality[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    NB_model = GaussianNB()
    NB_model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = NB_model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s Naive Bayes Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))
'''


####################################################Decision Tree
'''
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    # Let's train type indicator individually
    Y = list_personality[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    dtree = DecisionTreeClassifier(max_depth=5)
    dtree = dtree.fit(X_train, y_train)

    # make predictions for test data
    y_pred = dtree.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s Decision Tree Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))
'''


############################################################################Test
# A few few tweets and blog post

def jobRecommendation(personalityType):
    jobs = []
    if personalityType == "ISTJ":
        jobs.append("Dentist")
        jobs.append("Certified public accountant")
        jobs.append("Supply chain manager")
        jobs.append("Business analyst")

    elif personalityType == "INFJ":
        jobs.append("Counselor")
        jobs.append("Writer")
        jobs.append("Scientist")
        jobs.append("Psychologist")

    elif personalityType == "INTJ":
        jobs.append("Musical performer")
        jobs.append("Photographer")
        jobs.append("Financial advisor")
        jobs.append("Teacher")

    elif personalityType == "ENFJ":
        jobs.append("HR director")
        jobs.append("Public relations manager")
        jobs.append("Sales manager")
        jobs.append("Guidance counselor")

    elif personalityType == "ISTP":
        jobs.append("Technician")
        jobs.append("Engineer")
        jobs.append("Forensic scientist")
        jobs.append("Inspector")

    elif personalityType == "ESFJ":
        jobs.append("Office manager")
        jobs.append("Technical support specialist")
        jobs.append("Medical researcher")
        jobs.append("Psychologist")

    elif personalityType == "INFP":
        jobs.append("Copywriter")
        jobs.append("HR manager")
        jobs.append("Physical therapist")
        jobs.append("Artist")

    elif personalityType == "ESFP":
        jobs.append("Event planner")
        jobs.append("Sales representative")
        jobs.append("Tour guide")
        jobs.append("Flight attendant")

    elif personalityType == "ENFP":
        jobs.append("Editor")
        jobs.append("Musician")
        jobs.append("Product manager")
        jobs.append("Personal trainer")

    elif personalityType == "ESTP":
        jobs.append("Firefighter")
        jobs.append("Paramedic")
        jobs.append("Creative director")
        jobs.append("Project coordinator")

    elif personalityType == "ESTJ":
        jobs.append("Judge")
        jobs.append("Coach")
        jobs.append("Financial officer")
        jobs.append("Hotel manager")

    elif personalityType == "ENTJ":
        jobs.append("Business administrator")
        jobs.append("Public relations specialist")
        jobs.append("Mechanical engineer")
        jobs.append("Construction manager")

    elif personalityType == "INTP":
        jobs.append("Professor")
        jobs.append("Producer")
        jobs.append("Writer")
        jobs.append("Web developer")

    elif personalityType == "ISFJ":
        jobs.append("Judge")
        jobs.append("Coach")
        jobs.append("Financial officer")
        jobs.append("Hotel manager")

    elif personalityType == "ENTP":
        jobs.append("Event planner")
        jobs.append("Sales representative")
        jobs.append("Tour guide")
        jobs.append("Flight attendant")

    elif personalityType == "ISFP":
        jobs.append("Editor")
        jobs.append("Musician")
        jobs.append("Product manager")
        jobs.append("Personal trainer")
    return jobs

def validate(my_posts):
    mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

    my_posts, dummy = pre_process_data(mydata, remove_stop_words=True)

    my_X_cnt = cntizer.transform(my_posts)
    my_X_tfidf = tfizer.transform(my_X_cnt).toarray()

    # setup parameters for xgboost
    param = {}
    param['n_estimators'] = 200
    param['max_depth'] = 2
    param['nthread'] = 8
    param['learning_rate'] = 0.2

    result = []
    # Let's train type indicator individually
    for l in range(len(type_indicators)):
        #print("%s ..." % (type_indicators[l]))

        Y = list_personality[:, l]

        # split data into train and test sets
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        # fit model on training data
        model = XGBClassifier(**param, eval_metric='logloss')
        model.fit(X_train, y_train)

        # make predictions for my  data
        y_pred = model.predict(my_X_tfidf)
        result.append(y_pred[0])
    jobs = []
    jobs = jobRecommendation(translate_back(result))
    #print("Jobs: ", jobs)
    #print("results: ", translate_back(result))
    #print("posts: ", my_posts)
    return translate_back(result), jobs
    #print("The result is: ", translate_back(result))



'''
my_posts  = """Getting started with data science and applying machine learning has never been as simple as it is now. There are many free and paid online tutorials and courses out there to help you to get started. I’ve recently started to learn, play, and work on Data Science & Machine Learning on Kaggle.com. In this brief post, I’d like to share my experience with the Kaggle Python Docker image, which simplifies the Data Scientist’s life.
Awesome #AWS monitoring introduction.
HPE Software (now @MicroFocusSW) won the platinum reader's choice #ITAWARDS 2017 in the new category #CloudMonitoring
Certified as AWS Certified Solutions Architect 
Hi, please have a look at my Udacity interview about online learning and machine learning,
Very interesting to see the  lessons learnt during the HP Operations Orchestration to CloudSlang journey. http://bit.ly/1Xo41ci 
I came across a post on devopsdigest.com and need your input: “70% DevOps organizations Unhappy with DevOps Monitoring Tools”
In a similar investigation I found out that many DevOps organizations use several monitoring tools in parallel. Senu, Nagios, LogStach and SaaS offerings such as DataDog or SignalFX to name a few. However, one element is missing: Consolidation of alerts and status in a single pane of glass, which enables fast remediation of application and infrastructure uptime and performance issues.
Sure, there are commercial tools on the market for exactly this use case but these tools are not necessarily optimized for DevOps.
So, here my question to you: In your DevOps project, have you encountered that the lack of consolidation of alerts and status is a real issue? If yes, how did you approach the problem? Or is an ChatOps approach just right?
You will probably hear more and more about ChatOps - at conferences, DevOps meet-ups or simply from your co-worker at the coffee station. ChatOps is a term and concept coined by GitHub. It's about the conversation-driven development, automation, and operations.
Now the question is: why and how would I, as an ops-focused engineer, implement and use ChatOps in my organization? The next question then is: How to include my tools into the chat conversation?
Let’s begin by having a look at a use case. The Closed Looped Incidents Process (CLIP) can be rejuvenated with ChatOps. The work from the incident detection runs through monitoring until the resolution of issues in your application or infrastructure can be accelerated with improved, cross-team communication and collaboration.
In this blog post, I am going to describe and share my experience with deploying HP Operations Manager i 10.0 (OMi) on HP Helion Public Cloud. An Infrastructure as a Service platform such as HP Helion Public Cloud Compute is a great place to quickly spin-up a Linux server and install HP Operations Manager i for various use scenarios. An example of a good use case is monitoring workloads across public clouds such as AWS and Azure.
"""
'''
'''
my_posts = "Graffiti- CAS This was one of the best experience I have had till date in IB as I always wondered how people managed to create such amazing art works on wall in such a short time with so much ease. This was when I actually saw it with my own eyes. I had never expected to get an exposure as such. During this experience, we were taught a little about what graffiti is and how it evolved to be the way it is now. This was something I didn’t know so it really caught my attention.||| Getting to know that graffiti doesn’t need to be the wonderful art pieces on the wall as it had evolved from when cave man wrote on walls to writing your name on the wall to present where there is a mixture of drawings, writings, shapes and designs. After that we also learnt how to use the spray cans and different types of spray caps which are used for different purposes. We got the opportunity to work a lot on the designing of the art piece with different spray colors.||| After the art piece was done, we learnt how to write letters using markers on paper and different ways of how to hold the pen in order to get different effects of it. As I have always been interested in art, it was an experience I was really excited about and wanted to make the most of it.||| It made me like arts and designs more as I wasn’t able to take it as a subject, so it made me feel free and able to express myself after a very long time. The people who came to teach us, helped us throughout the experience by explaining on how to use the different techniques as well as helped us write our names in different fonts.||| I finally feel that if I practice a little more, I will be able to make different art pieces using the techniques they have taught us. Describe a topic idea or concept you find so engaging that it makes you lose all track of time why does it captivate you? What or who do you turn to when you want to learn more. Death or life after death has always been a topic that mesmerizes me as well as shakes me up a little from the inside knowing that the person that was the most closest to you and someone you rely one could leave you at any second of life or even you could leave them.||| Although death being such a common things, no one knows how it happens of what it feels like. There are several myths that we hear about death but no one actually know what it is or how it feels. Every day we hear that thousands of people die in a day and we cry for days, weeks, months or maybe years also but we never know where or how that person is or whether or not that person has reached something that we call heaven.||| We all assume that heaven is the place where a soul meets up with God while hell being a place filled with fire or other things we associate with the devil. Many people and religions believe that after we die, we meet angels or even when a person is going to die, we usually say the death angel took his/ her soul. But to what extent is this true and do angels and devils actually exist? Many religions believe in different processes that take place after a person dies but surprisingly the different myths about life after death, as we are all human beings, doesn’t that mean we all take the same route/ path after a person dies?||| This has been a question that I guess no one is and will be able to answer as there is no one who has had an experience of such and could tell us about it. Talking to different people about death always arises more questions that might not be answered by anyone but due to different beliefs and understanding, we may come up with several answers to them.||| Many people may die a regular death or due to sicknesses or maybe an accident while others may commit suicide or being murdered. As mentioned there are several reasons to why an individual could die, but still people believe that a person can only die when God has written it is time for him/her to die.||| Does that mean the person who commits suicide or dies cause of a murder would have known the exact time of when he or she would die? And also after a person dies, every religion has different ways of caring out the funeral process in which some people burn the body while others bury.||| This makes me wonder why different ways types of funeral if both the souls will be going to the same God or path. Usually when I come up with such questions or want to learn more about it, I discuss it with my family and friends during certain gather ups. Here I am able to get different perspectives as well as a better understanding of what has been taught to us not only in our religion but also what our elders have learnt from their teachers, friends and family. This allows me to broaden my way of thinking as well as become an inquirer and find out more not only about this topic but other topics as well that may rise up from this particular topic.||| Share an essay on any topic of your choice it can be one you have already written one that describes to a different prompt or one of your own strength. Starting from a childhood that never had friends to a teenage life where friends are all around you at all times has been my journey of friendship. Never would I have imagined living a life where I am surrounded not only by people I want to be around with, but also people that I don’t enjoy being with.||| This was one of the greatest teachings I have learnt from life so far. Living my childhood in a small town in Tanzania called Iringa, where I had no friends of my age and gender. I had a tendency of either being alone at home and taking care of my sisters or either hanging around with my male cousins who are about 7 years older than me playing football during our free times. As simple as life felt, I had never felt the need to have any other friends or people around me till the time I moved to gurukul. A place where I start my day off with a friend waking me up, to the time where I say the final words of the day to my roommates with a goodnight. Living a life in hostel has made a major impact in my social life, allowing me to interact with different types of people at various times of my life.||| This may include people that I get along with such as my best friends but also other people who I might have had arguments and fights with or people that just don’t seem like the people I would want to be around. This made understand that in the future we don’t have a choice as to the type of people we want to be around and we have to manage being around different people.||| As per my future plans, I would like to be a nurse and therefore, interacting with different people is a major aspect that according to me I had to learn as I lacked a lot in the section."

# The type is just a dummy so that the data prep function can be reused
mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

my_posts, dummy  = pre_process_data(mydata, remove_stop_words=True)

my_X_cnt = cntizer.transform(my_posts)
my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()

# setup parameters for xgboost
param = {}
param['n_estimators'] = 200
param['max_depth'] = 2
param['nthread'] = 8
param['learning_rate'] = 0.2

result = []
# Let's train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    Y = list_personality[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier(**param, eval_metric='logloss')
    model.fit(X_train, y_train)

    # make predictions for my  data
    y_pred = model.predict(my_X_tfidf)
    result.append(y_pred[0])
print("The result is: ", translate_back(result))

'''




