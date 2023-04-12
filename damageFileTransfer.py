import os
import shutil
import pandas as pd

csv_path = r'Redacted'

dir_path = r'Redacted'

df = pd.read_csv(csv_path)

categories = df['classes'].unique()
print(len(categories))

for category in categories:
    os.makedirs(os.path.join(dir_path, category), exist_ok=True)

for index, row in df.iterrows():
    filename = row['image'].split('/')[-1]
    category = row['classes']
    src_path = os.path.join(r'Redacted', filename)
    dst_path = os.path.join(dir_path, category, filename)
    shutil.copyfile(src_path, dst_path)
