'''
Created on Apr 20, 2018

@author: nishant.sethi
'''
import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.svm import SVC

match_data=pd.read_csv('matches.csv')
print(match_data.info())
match_data=match_data.drop('umpire3',axis=1)
print("***************************************************")
match_data.info()
null_cities=match_data[match_data['city'].isna()]
print("*************************************************** data with cities as null values ***************************************************")
print(null_cities)
print("***************************************************")
null_umpire1=match_data[match_data['umpire1'].isna()]
print("data with umpire1 as null values")
print(null_umpire1)
print("***************************************************8")
null_umpire2=match_data[match_data['umpire2'].isna()]
print("data with umpire2 as null values")
print(null_umpire2)
x=match_data.iloc[4]
print("row 5")
print(match_data.iloc[4]['umpire1'])
x=match_data[match_data['umpire1'].isna()]['id']
print(match_data.ix[x-1,'umpire1'])
match_data.ix[x-1,'umpire1']='Virender Sharma'
print(match_data.ix[x-1,'umpire2'])
x=match_data[match_data['umpire2'].isna()]['id']
match_data.ix[x-1,'umpire2']='Sundaram Ravi'
print("*"*100)
print("after filling null values of umpire1 and umpir2")
match_data.info()
# print("*"*100)
# print("ids of matches where cities is null")
# x=match_data[match_data['city'].isna()]['id']
# print("x",x)
# match_data=match_data.drop(match_data[match_data['city'].isna()]['id'])
# print("*"*100)
# print("data info after dropping cities with values equal to null")
# match_data.info()
# print(list(x))
print("*"*100)
player_of_the_match_null=match_data[match_data['player_of_match'].isna()]
print("data with player_of_the_match as null values")
print(player_of_the_match_null)
x=match_data[match_data['player_of_match'].isna()]['id']
for i in x:
    match_data.ix[i,'player_of_match']='no result'
x=match_data[match_data['winner'].isna()]['id']
for i in x:
    match_data.ix[i,'winner']='no result'
print("*"*100)
print("after filling null values of winner and player of the match")
print("data info:")
match_data.info()

x=match_data[match_data['city'].isna()]['id']
for i in x:
    match_data.ix[i,'city']='NULL'
print("*"*100)
print("after filling null values of cities")
match_data.info()
print(match_data.describe())




print("point table of year 2015")
match_data_group=match_data[match_data['season']==2015].groupby('winner')['winner'].count()
print((match_data_group))


print("no of times team won by batting first")
match_data_group=match_data[match_data['toss_decision']=='bat'].groupby('winner')['winner'].count()
print((match_data_group))

print("no of times team won by batting second")
match_data_group=match_data[match_data['toss_decision']=='field'].groupby('winner')['winner'].count()
print((match_data_group))

print("no of times team won by winning the toss")
x=match_data[match_data['toss_winner']==match_data['winner']]
match_data_group=x.groupby('winner')['winner'].count()
print(match_data_group)
 
print("chennai status")
face_off=['Chennai Super Kings','Rajasthan Royals']
team="Chennai Super Kings"
x=match_data[((match_data['team1']=='Kolkata Knight Riders')|(match_data['team2']=='Kolkata Knight Riders'))]
match_data_group=x.groupby('winner')['winner'].count()
print(match_data_group)
x=match_data[((match_data['team1']=='Kings XI Punjab')|(match_data['team2']=='Kings XI Punjab'))]
match_data_group=x.groupby('winner')['winner'].count()
print(match_data_group)







x=(match_data['team1']=='Kolkata Knight Riders') & (match_data['team2']=='Kings XI Punjab')
y=(match_data['team2']=='Kolkata Knight Riders') & (match_data['team1']=='Kings XI Punjab')
today_match_data=match_data[x | y]
print(today_match_data)

today_match_data['city']=today_match_data['city'].map({'Kolkata':0,'Cuttack':0,'Chandigarh':1,'Durban':3,'Port Elizabeth':3,'Abu Dhabi':3,'Bangalore':3,'Pune':3})
today_match_data['team1']=today_match_data['team1'].map({'Kolkata Knight Riders':0,'Kings XI Punjab':1})
today_match_data['team2']=today_match_data['team2'].map({'Kolkata Knight Riders':0,'Kings XI Punjab':1})
today_match_data['toss_winner']=today_match_data['toss_winner'].map({'Kolkata Knight Riders':0,'Kings XI Punjab':1})
today_match_data['toss_decision']=today_match_data['toss_decision'].map({'bat':0,'field':1})
today_match_data['winner']=today_match_data['winner'].map({'Kolkata Knight Riders':0,'Kings XI Punjab':1})
today_match_data=today_match_data.drop(['id','date','season','result','dl_applied','win_by_wickets', 'player_of_match','venue','umpire1','umpire2','win_by_runs','team1','team2'],axis=1)
a=today_match_data.columns
print(a)
print(today_match_data)
today_match_data.to_csv("C:\\Users\\nishant.sethi\\Desktop\\today_match_data.csv")

output_data=today_match_data['winner']
training_data=today_match_data.drop('winner',axis=1)

print(training_data)
print(output_data)

X=np.array(training_data)
y=np.array(output_data)

svc=SVC()
svc.fit(X,y)
test=np.array([0,1,1]).reshape(1,-1) #'''rajasthan win and bat'''
print(svc.predict(test))
test=np.array([3,1,1]).reshape(1,-1) #'''rajasthan win and bowl'''
print(svc.predict(test))
test=np.array([3,0,0]).reshape(1,-1) #'''chennai win and bat'''
print(svc.predict(test))
test=np.array([3,0,1]).reshape(1,-1) #'''chennai win and bowl'''
print(svc.predict(test))


matches = pd.read_csv('matches.csv')
matches["type"] = "pre-qualifier"
for year in range(2008, 2017):
    final_match_index = matches[matches['season']==year][-1:].index.values[0]
    matches["type"][final_match_index]="final"
    matches["type"][final_match_index-1]="qualifier-2"
    matches["type"][final_match_index-2]="eliminator"
    matches["type"][final_match_index-3]="qualifier-1"

matches.groupby(["type"])["id"].count()
print(matches.head())




























