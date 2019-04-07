import pandas as pd 
import json 
import os 

with open('annotations.json') as f: 
  ann = json.load(f) 

# Label groups
label_dict = {}
for label in ann['labelGroups'][0]['labels']: 
  label_dict[label['id']] = label['name']

# Abnormals
keys = ['SOPInstanceUID', 'StudyInstanceUID', 'SeriesInstanceUID', 'labelId']
annot_dict = {k : [] for k in keys}

for annot in ann['datasets'][0]['annotations']: 
  for k in keys:
    if k in annot.keys():
      annot_dict[k].append(annot[k])
    else:
      annot_dict[k].append(None)

abnormal_annot_df = pd.DataFrame(annot_dict)
abnormal_annot_df['labelId'] = [label_dict[_] for _ in abnormal_annot_df['labelId']]

# Normals
annot_dict = {k : [] for k in keys}

for annot in ann['datasets'][1]['annotations']: 
  for k in keys:
    if k in annot.keys():
      annot_dict[k].append(annot[k])
    else:
      annot_dict[k].append(None)

normal_annot_df = pd.DataFrame(annot_dict)
normal_annot_df['labelId'] = [label_dict[_] for _ in normal_annot_df['labelId']]

annot_df = pd.concat([abnormal_annot_df, normal_annot_df])

acl_df = annot_df[annot_df['labelId'].isin(['ACL Tear', 'Normal ACL'])]

# Get files
filenames = [] 
for root, dirs, files in os.walk('data'): 
  for fi in files:
     filenames.append(os.path.join(root, fi))

files_df = pd.DataFrame({'filepath': filenames}) 
files_df['SOPInstanceUID'] = [_.split('/')[-1] for _ in files_df['filepath']]

acl_df = acl_df.merge(files_df, on='SOPInstanceUID')

# Set up folds
cv_df = acl_df[['StudyInstanceUID', 'labelId']].drop_duplicates()
n_test_folds = 10
n_val_folds  = 8

cv_df['fold'] = 88888
for i in range(n_val_folds): 
  cv_df['val{}'.format(i)] = 'test'


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=n_test_folds, shuffle=True, random_state=0)

fold_counter = 0
for train_index, test_index in skf.split(cv_df, cv_df['labelId']):
  train_df = cv_df.iloc[train_index] 
  test_df  = cv_df.iloc[test_index]
  cv_df['fold'].iloc[test_index] = fold_counter
  valid_fold_counter = 0 
  train_skf = StratifiedKFold(n_splits=n_val_folds, shuffle=True, random_state=0)
  for train_index, valid_index in train_skf.split(train_df, train_df['labelId']):
    cv_df['val{}'.format(valid_fold_counter)].iloc[valid_index] = 'valid'
    cv_df['val{}'.format(valid_fold_counter)].iloc[train_index] = 'train'
    valid_fold_counter += 1
  fold_counter += 1

del cv_df['labelId']
acl_df = acl_df.merge(cv_df, on='StudyInstanceUID')

acl_df.to_csv('acl_df_splits.csv', index=False)







