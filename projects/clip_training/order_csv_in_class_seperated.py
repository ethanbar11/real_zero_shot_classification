 # DONT USE THIS
 # DONT USE THIS
 # DONT USE THIS
 # DONT USE THIS
 # DONT USE THIS
 # DONT USE THIS
 # DONT USE THIS
 # DONT USE THIS
 # DONT USE THIS
 # DONT USE THIS
 # DONT USE THIS



import pandas as pd

path = r'tmp/train.csv'
df = pd.read_csv(path)
# Assuming df is your original dataframe
groups = df.groupby('class_name')

# Create a new dataframe
# new_df = pd.DataFrame(columns=['image_path', 'class_name', 'sentence'])

max_items = len(df)
group_indices = {name: 0 for name in groups.groups}  # Tracks the current index for each group
added_items = 0
lst = []
# Keep iterating until you have added all items from the original dataframe
should_run = True
while added_items < max_items and should_run:
    for name, group in groups:
        if group_indices[name] < len(group):  # Check if the current group still has items left
            # Get the row from the group
            row = group.iloc[group_indices[name]]
            # Add to the new dataframe
            lst.append(
                ({'img_path': row['img_path'], 'class_name': row['class_name'], 'sentence': row['sentence']}))
            added_items += 1
            group_indices[name] += 1  # Increment the index for the current group
        else:
            should_run = False

        if added_items >= max_items:  # Break if all items are added
            break

new_df = pd.DataFrame(lst)
print(new_df.loc[:, 'class_name'])
new_df.to_csv('tmp/train_ordered.csv', index=False)
