import warnings

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample


warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set(style='white', context='notebook', palette='deep')

data_set = pd.read_csv("../dataset/p0_q0.3.csv")
salaries = [50000]

# кореляция численных величин

img = sns.heatmap(data_set[['salary', 'gender', 'age', 'city', 'education', 'work_exp', 'position', 'fields_of_activity', 'employment', 'work_schedule']].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.savefig("../dataset_analyze/0.png", figsize=(1000,1000))
plt.clf()
exit(0)
print('Rows count: ', len(data_set))

# удаление эмбедингов
'''
i = 0
while i < 100:
    data_set.drop(labels=[str(i)], axis=1, inplace=True)
    i = i + 1
'''
# удаление остальных данных
'''
data_set.drop(labels=['gender','age','city','education','position','fields_of_activity','employment','work_schedule','friends_count','friends_median_age','friends_mid_age','groups_count'], axis=1, inplace=True)
'''
# удаление величин не коррелирующих с зарплатой
data_set.drop(labels=['fields_of_activity','employment','friends_count','friends_median_age','friends_mid_age','groups_count'], axis=1, inplace=True)
# замена значений зарплат

def map_salary(salary):
    i = 0

    while i < len(salaries):
        if salary <= salaries[i]:
            break
        i = i + 1

    return i

data_set['salary'] = data_set['salary'].apply(map_salary)

# разделение данных на обучающее и тестовое множества

validation_size = 0.2
seed = 7
num_folds = 10
scoring = 'accuracy'

train, test = train_test_split(data_set, test_size=validation_size, random_state=seed)

frames = list()
vc = train['salary'].value_counts()

for i in data_set['salary'].unique():
    if vc[i] < vc.max():
        frames.append(resample(train[train['salary'] == i], replace = True, n_samples = vc.max()))
    else:
        frames.append(train[train['salary'] == i])

train = pd.concat(frames)

# обучение

train_array = train.values

x_t = train_array[:, 1:]
y_t = train_array[:, 0]

test_array = test.values

x_v = test_array[:, 1:]
y_v = test_array[:, 0]

print('Start train')

random_forest = RandomForestClassifier(n_estimators=100, max_features=None, max_depth=None, oob_score=True)
random_forest.fit(x_t, y_t)

# оценка результатов

predictions = random_forest.predict(x_v)

print("Accuracy: %s%%" % (100 * accuracy_score(y_v, predictions)))
print()
print(confusion_matrix(y_v, predictions))
print()
print(classification_report(y_v, predictions))
