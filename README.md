# MAIS-Hacks-2021
#Inspiration

  NLP has always been an important filed of Machine Learning and AI, which can fill the gap between human communication and computer understanding. As a team with students minoring in Psychology and Linguistics, we know how much power language possesses and how important it is to try to keep a healthy mood. That's why some of us keep a diary.
  But can we do something to break the barrier of the cold User Interface between a diary app and us? To make the machine truly understand what we are feeling right now?
  With this inspiration, in this project, we take a step further to explore how to predicate the emotion of words in a precise and concise way. This can be expanded to a diary app with mood tracking and analyzing functions.

#What it does

  Give a predication of user's emotion based on their text input.

#How we built it

  First, we found a lot of datasets containing tweets, sentences, movie scripts with labelled emotions and preprocessed them to cater for the training of the Mood Tracker.
  Once we had the merged and preprocessed dataset, we tried several different machine learning models. The model with the best performance was SVM.
  In the end, using Flask, we built a simple landing page webapp and deployed via Heroku which is the app’s first step towards a friendly user interface.

#Challenges we ran into

  Hard to clean the data as most of the tweets are malformatted.
  Difficult to label the sentence as emotions cannot be defined with high precision. Mixed emotions exist.
  Not able to further improve the performance of the model.
  Deploying to Heroku caused a lot of unexpected errors.
  Accomplishments that we're proud of
  First and foremost, the fact that it works!! None of us had much experience with NLP, so it was a great NLP learning experience.
  During the process, we learned a lot about data preprocessing and familiarized ourselves with different machine learning models.
  The landing page -as simple as it is- was also a great accomplishment because we aimed to have a user friendly interface and that is a first step towards that.
 
 #What we learned
 
  How to apply NLP algorithms and models to a practical degree
  How to share code and split tasks effectively
  How bad the format of the most Tweets are

#What's next for Mood Tracker

  Due to time constraints, we could not experiment with as many machine learning models as we wished. We also could not fine tune much. Experimenting with different models and fine tuning is definitely a next step.
  Since our dataset mostly contained Tweets and short sentences, we are not so sure how accurate the model will be for longer inputs. Testing that and trying to find some datasets with longer inputs would be interesting.
  A more friendly user-interface and a more aesthetic landing-page.
  Maybe expand it to an actual app so we could make money
  
#Built With

  flask   nltk  numpy   pandas    python    scikit-learn
  
#Try it out

https://moodtrackerhack2.herokuapp.com/
