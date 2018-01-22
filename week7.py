import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn import metrics # for scoring = "accuracy"
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# features=["match_id","start_time","lobby_type","r1_hero","r1_level","r1_xp","r1_gold","r1_lh","r1_kills","r1_deaths","r1_items","r2_hero","r2_level","r2_xp","r2_gold","r2_lh","r2_kills","r2_deaths","r2_items","r3_hero","r3_level","r3_xp","r3_gold","r3_lh","r3_kills","r3_deaths","r3_items","r4_hero","r4_level","r4_xp","r4_gold","r4_lh","r4_kills","r4_deaths","r4_items","r5_hero","r5_level","r5_xp","r5_gold","r5_lh","r5_kills","r5_deaths","r5_items","d1_hero","d1_level","d1_xp","d1_gold","d1_lh","d1_kills","d1_deaths","d1_items","d2_hero","d2_level","d2_xp","d2_gold","d2_lh","d2_kills","d2_deaths","d2_items","d3_hero","d3_level","d3_xp","d3_gold","d3_lh","d3_kills","d3_deaths","d3_items","d4_hero","d4_level","d4_xp","d4_gold","d4_lh","d4_kills","d4_deaths","d4_items","d5_hero","d5_level","d5_xp","d5_gold","d5_lh","d5_kills","d5_deaths","d5_items","first_blood_time","first_blood_team","first_blood_player1","first_blood_player2","radiant_bottle_time","radiant_courier_time","radiant_flying_courier_time","radiant_tpscroll_count","radiant_boots_count","radiant_ward_observer_count","radiant_ward_sentry_count","radiant_first_ward_time","dire_bottle_time","dire_courier_time","dire_flying_courier_time","dire_tpscroll_count","dire_boots_count","dire_ward_observer_count","dire_ward_sentry_count","dire_first_ward_time","duration","radiant_win","tower_status_radiant","tower_status_dire","barracks_status_radiant","barracks_status_dire"]

# features_test=["match_id","start_time","lobby_type","r1_hero","r1_level","r1_xp","r1_gold","r1_lh","r1_kills","r1_deaths","r1_items","r2_hero","r2_level","r2_xp","r2_gold","r2_lh","r2_kills","r2_deaths","r2_items","r3_hero","r3_level","r3_xp","r3_gold","r3_lh","r3_kills","r3_deaths","r3_items","r4_hero","r4_level","r4_xp","r4_gold","r4_lh","r4_kills","r4_deaths","r4_items","r5_hero","r5_level","r5_xp","r5_gold","r5_lh","r5_kills","r5_deaths","r5_items","d1_hero","d1_level","d1_xp","d1_gold","d1_lh","d1_kills","d1_deaths","d1_items","d2_hero","d2_level","d2_xp","d2_gold","d2_lh","d2_kills","d2_deaths","d2_items","d3_hero","d3_level","d3_xp","d3_gold","d3_lh","d3_kills","d3_deaths","d3_items","d4_hero","d4_level","d4_xp","d4_gold","d4_lh","d4_kills","d4_deaths","d4_items","d5_hero","d5_level","d5_xp","d5_gold","d5_lh","d5_kills","d5_deaths","d5_items","first_blood_time","first_blood_team","first_blood_player1","first_blood_player2","radiant_bottle_time","radiant_courier_time","radiant_flying_courier_time","radiant_tpscroll_count","radiant_boots_count","radiant_ward_observer_count","radiant_ward_sentry_count","radiant_first_ward_time","dire_bottle_time","dire_courier_time","dire_flying_courier_time","dire_tpscroll_count","dire_boots_count","dire_ward_observer_count","dire_ward_sentry_count","dire_first_ward_time"]

data = pandas.read_csv('./features.csv', index_col='match_id')
data_test = pandas.read_csv('./features_test.csv', index_col='match_id')
features = data.columns
features_test = data_test.columns


# 1. Какие признаки имеют пропуски среди своих значений? Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?

Count = data.count()
Nan = Count[Count < data.shape[0]]
print(Nan)

'''
Перечисленные ниже 12 признаков имеют пропуски. Судя по фрагментам названий "first" и "time", 
речь идет о моментах времени, в которые происходили некоторые события. Если события 
не происходили,то и соответствующие поля пустые.

first_blood_time               77677
first_blood_team               77677
first_blood_player1            77677
first_blood_player2            53243
radiant_bottle_time            81539
radiant_courier_time           96538
radiant_flying_courier_time    69751
radiant_first_ward_time        95394
dire_bottle_time               81087
dire_courier_time              96554
dire_flying_courier_time       71132
dire_first_ward_time           95404
'''

data.fillna(0) # заполним пропуски в данных нулями, согласно указаниям в задании

X = data[features_test]
y = data['radiant_win']

df = pandas.DataFrame([[np.nan, 2, np.nan, 0],
                    [3, 4, np.nan, 1],
                    [np.nan, np.nan, np.nan, 5],
                    [np.nan, 3, np.nan, 4]],
                    columns=list('ABCD'))
