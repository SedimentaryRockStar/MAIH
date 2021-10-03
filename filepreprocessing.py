import csv
import pandas as pd
import os
import re
from nltk.tokenize import TweetTokenizer

directory1 = './rawData/Data1'
directory2 = './rawData/Data2'
directory3 = './rawData/Data3'
directory4 = './rawData/Data4'

DataList1 = []  # Initialize a new list
DataList2 = []
DataList3 = []
DataList4 = []


# Remove the emojis that could contaminate the data
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)   # @ saaranshM on github


for (root,dirs,file) in os.walk(directory1, topdown=True):
    for idx in range(len(file)):
        if file[idx]== '.DS_Store':
            continue
        csv_file = open(directory1+'/'+ file[idx], 'r')
        csv_reader = csv.reader(csv_file)
        column= next(csv_reader)  # skip column names
        DataList1 += list(csv_reader)


for (root,dirs,file) in os.walk(directory2, topdown=True):
    for idx in range(len(file)):
        if file[idx]== '.DS_Store':
            continue
        f = open(directory2+'/'+ file[idx], 'r')
        for line in f:
            token= remove_emoji(line).split("\t")
            DataList2.append([token[2], token[1]])
        f.close()



for (root,dirs,file) in os.walk(directory3, topdown=True):
    for idx in range(len(file)):
        if file[idx]== '.DS_Store':
            continue
        csv_file = open(directory3 + '/' + file[idx], 'r')
        csv_reader = csv.reader(csv_file)
        column = next(csv_reader)  # skip column names
        for line in csv_reader:
            DataList3.append([line[1], line[2]])

print(len(DataList3)+ len(DataList1)+ len(DataList2))

for (root,dirs,file) in os.walk(directory4, topdown=True):
    for idx in range(len(file)):
        if file[idx]== '.DS_Store':
            continue
        f = open(directory4+'/'+ file[idx], 'r')
        for line in f:
            token= remove_emoji(line).split(";")
            DataList4.append([token[1].strip('\n'), token[0]])
        f.close()
print(len(DataList4))

df = pd.DataFrame(DataList1+ DataList2+ DataList3, columns=["Emotion", "Sentence"])
df.to_csv("mergedData1.csv", index=False)

df = pd.DataFrame(DataList4, columns=["Emotion", "Sentence"])
df.to_csv("mergedData2.csv", index=False)







f = open('./rawData/mergedData3_delete_empty.csv', 'r', encoding="latin-1")
csv_reader = csv.reader(f)
allowed = "QWERTYUIOPASDFGHJKLZXCVBNMqwertyuopasdfghjklizxcvbnm "

operated= []
tweet = TweetTokenizer()
column = next(csv_reader)


for line in csv_reader:

    aString = []
    for idx in range(len(line[1])):
        if line[1][idx] == " " and idx - 1 >= 0 and idx + 1 < len(line[1]) and line[1][idx - 1] == 'n' and line[1][idx + 1] == 't':
            continue
        if line[1][idx]== chr(39):
            if idx-2>= 0 and line[1][idx- 2]== " " and idx- 1>= 0 and line[1][idx- 1]== 'n':
                c= aString.pop()
                aString.pop()
                aString.append(c) # Deal with coud n't
            elif idx- 1>= 0 and line[1][idx- 1]== " ":
                aString.pop() # Deal with I 'm
        else:
            aString.append(line[1][idx])


    newString = ""
    for ele in aString:
        newString += ele
    line[1] = newString

    raw= tweet.tokenize(line[1])
    string= ""
    for ele in raw:
        for char in ele:
            if char not in allowed:
                ele= ele. replace(char, "")
        if ele != "":
            string+= ele+ " "
    string= string.strip()
    operated.append([line[0], string])


df = pd.DataFrame(operated, columns=["Emotion", "Sentence"])
df.to_csv("mergedData4_delete_empty.csv", index=False)