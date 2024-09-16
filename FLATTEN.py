import os

import pandas as pd
import pm4py
from sklearn.preprocessing import LabelEncoder

ocel_name = "order-management"
output_folder = os.path.join('.', 'data', ocel_name)

os.makedirs(output_folder, exist_ok=True)

ocel = pm4py.read.read_ocel2_json(os.path.join('.', 'data', f'{ocel_name}.json'))

#ocel = pm4py.read.read_ocel(os.path.join('.', 'data', f'{ocel_name}.jsonocel'))

otypes = pm4py.ocel.ocel_get_object_types(ocel)

otypes_to_combine = ['orders', 'items', 'packages']
# List to store all DataFrames
all_dfs = []

# Define the desired column order
desired_order = ['CaseID', 'ActivityName', 'timestamp', 'ObjectType']

label_encoder = LabelEncoder()

all_activities = set()

for ot in otypes:
    pd_df = pm4py.ocel.ocel_flattening(ocel, ot)
    pd_df = pd_df.rename(
        columns={'concept:name': 'ActivityName', 'case:concept:name': 'CaseID', 'time:timestamp': 'timestamp'})

    # Add the activity names to the set
    all_activities.update(pd_df['ActivityName'].unique())

# Fit the LabelEncoder on all unique activities across all object types
label_encoder.fit(list(all_activities))

for ot in otypes:
    pd_df = pm4py.ocel.ocel_flattening(ocel, ot)

    pd_df = pd_df.rename(
        columns={'concept:name': 'ActivityName', 'case:concept:name': 'CaseID', 'time:timestamp': 'timestamp'})

    pd_df['ActivityID'] = label_encoder.transform(pd_df['ActivityName'])

    # Reorder columns
    columns = [col for col in desired_order if col in pd_df.columns] + [col for col in pd_df.columns if
                                                                        col not in desired_order]
    pd_df = pd_df[columns]

    # Save individual DataFrame
    pd_df.to_csv(os.path.join(output_folder, f'{ot}.csv'), index=False)

    # Append to the list
    if ot in otypes_to_combine:
        all_dfs.append(pd_df)
# Combine all DataFrames into one
combined_df = pd.concat(all_dfs, ignore_index=True)

# Reorder columns in the combined DataFrame
combined_columns = [col for col in desired_order if col in combined_df.columns] + [col for col in combined_df.columns if
                                                                                   col not in desired_order]
combined_df = combined_df[combined_columns]

# Save the combined DataFrame
combined_df.to_csv(os.path.join(output_folder, 'combined_flat_ocel.csv'), index=False)
