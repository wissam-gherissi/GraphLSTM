import os
import pickle

import pandas as pd
import numpy as np
import random

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm

from graph_helpers import get_dfg_from_df, unionize_dfg_sources, data_generator

ACTIVITY_KEY = 'ActivityID'
CASE_ID_KEY = 'CaseID'
TIMESTAMP_KEY = 'timestamp'



def preprocess_and_prepare_graphs(model_used, ocel, main_ot, ocdfg, threshold):
    # Read main CSV file
    main_file_path = os.path.join('.', 'data', ocel, f'{main_ot}.csv')
    # Load the CSV file into a DataFrame
    df_main = pd.read_csv(main_file_path)

    # Preprocess data into cases and calculate temporal features and number of related objects
    data_df = preprocess_data(df_main)

    # Split data into train and test sets
    train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=1)

    # train_cases = np.array(train_data['caseid'])
    # df1 = df_main[df_main['CaseID'].isin(train_cases)].copy()

    # Find the maximum line size
    maxlen = max(len(line) for line in data_df['lines'])

    # Prepare character encoding
    chars, char_indices, indices_char, char_act = prepare_character_encoding(data_df['lines'])

    train_sequences = construct_sequences(train_data)

    test_sequences = construct_sequences(test_data)

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Prepare LSTM input and targets for training data
    X_train_lstm, y_act_train, y_times_train, time_target_means = prepare_lstm_input_and_targets(
        train_sequences,
        maxlen,
        char_indices,
        char_act, scaler)

    # Prepare LSTM input and targets for test data
    X_test_lstm, y_act_test, y_times_test, _ = prepare_lstm_input_and_targets(test_sequences, maxlen, char_indices,
                                                                              char_act, scaler)
    training_input = []
    test_input = []
    if model_used == "LSTM":
        for i in range(len(X_train_lstm)):
            d = Data()
            d.lstm_input = torch.from_numpy(X_train_lstm[i]).unsqueeze(0)
            d.y_act = torch.from_numpy(y_act_train[i]).unsqueeze(0)
            d.y_times = torch.from_numpy(y_times_train[i]).unsqueeze(0)
            training_input.append(d)
        for i in range(len(X_test_lstm)):
            d = Data()
            d.lstm_input = torch.from_numpy(X_test_lstm[i]).unsqueeze(0)
            d.y_act = torch.from_numpy(y_act_test[i]).unsqueeze(0)
            d.y_times = torch.from_numpy(y_times_test[i]).unsqueeze(0)
            test_input.append(d)
        config = [maxlen, time_target_means, char_indices]

    elif model_used == "GRAPH":

        # Graph preprocessing
        print("Graph preprocessing...")
        dfg_sources = []  # (get_dfg_from_df(df1, activity_key=ACTIVITY_KEY, case_id_key=CASE_ID_KEY,
        #                                timestamp_key=TIMESTAMP_KEY), main_ot)]
        for i, object_name in enumerate(ocdfg):
            file_path = os.path.join('.', 'data', ocel, f'{object_name}.csv')
            # Load the CSV file into a DataFrame
            df_dfg = pd.read_csv(file_path)
            dfg_data = get_dfg_from_df(df_dfg, activity_key=ACTIVITY_KEY, case_id_key=CASE_ID_KEY,
                                       timestamp_key=TIMESTAMP_KEY)
            dfg_sources.append((dfg_data, object_name))

        # Unionize DFG sources
        G_union = unionize_dfg_sources(dfg_sources, threshold=threshold)
        # Add padding node
        # G_union.add_node(0)

        # Prepare training data
        onehot_encoder = OneHotEncoder()
        scaler = StandardScaler()
        print("Train data generation...")
        training_input, num_edge_features = data_generator(G_union, X_train_lstm, y_act_train, y_times_train,
                                                           onehot_encoder, scaler, training=True)
        print("Test data generation...")
        test_input, _ = data_generator(G_union, X_test_lstm, y_act_test, y_times_test, onehot_encoder, scaler,
                                       training=False)

        config = [maxlen, time_target_means, char_indices, num_edge_features]

    # Save training and test inputs to pickle files

    # Define the base directory for the pickle files
    pickle_dir = os.path.join('.', 'pickle_files')

    # Construct file paths using os.path.join()
    train_file_path = os.path.join(pickle_dir, f'trainset_{ocel}_{main_ot}_{model_used}_{threshold}.pkl')
    test_file_path = os.path.join(pickle_dir, f'testset_{ocel}_{main_ot}_{model_used}_{threshold}.pkl')
    config_file_path = os.path.join(pickle_dir, f'config_{ocel}_{main_ot}_{model_used}_{threshold}.pkl')

    # Save the training input
    with open(train_file_path, 'wb') as train_file:
        pickle.dump(training_input, train_file)

    # Save the test input
    with open(test_file_path, 'wb') as test_file:
        pickle.dump(test_input, test_file)

    # Save the configuration
    with open(config_file_path, 'wb') as config_file:
        pickle.dump(config, config_file)


def load_data(model_used, ocel, main_ot, threshold):
    pickle_dir = os.path.join('.', 'pickle_files')
    train_file_path = os.path.join(pickle_dir, f'trainset_{ocel}_{main_ot}_{model_used}_{threshold}.pkl')
    test_file_path = os.path.join(pickle_dir, f'testset_{ocel}_{main_ot}_{model_used}_{threshold}.pkl')
    config_file_path = os.path.join(pickle_dir, f'config_{ocel}_{main_ot}_{model_used}_{threshold}.pkl')

    with open(train_file_path, 'rb') as train_file:
        training_input = pickle.load(train_file)

    with open(test_file_path, 'rb') as test_file:
        test_input = pickle.load(test_file)

    with open(config_file_path, 'rb') as config_file:
        if model_used == "GRAPH":
            Maxlen, time_target_means, Char_indices, num_edge_features = pickle.load(config_file)
        elif model_used == "LSTM":
            Maxlen, time_target_means, Char_indices = pickle.load(config_file)
            num_edge_features = 0

    return training_input, test_input, Maxlen, time_target_means, Char_indices, num_edge_features


def preprocess_data(df):
    caseids, lines, timeseqs, timeseqs2, timeseqs3, timeseqs4, nb_itemseqs, timeseqsF = [], [], [], [], [], [], [], []

    df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')  # Convert timestamp column to datetime

    for case_id, group in tqdm(df.groupby('CaseID'), desc='Preprocessing data'):
        caseids.append(case_id)
        lines.append(''.join(chr(int(activity_id) + 64) for activity_id in group['ActivityID']))
        # Calculate time differences since last event
        times = [(group['timestamp'].iloc[i] - group['timestamp'].iloc[i - 1]).total_seconds()
                 for i in range(1, len(group))]
        times.insert(0, 0)  # Insert 0 for the first event
        timeseqs.append(times)
        # Calculate time differences since case start
        times2 = [(group['timestamp'].iloc[i] - group['timestamp'].iloc[0]).total_seconds()
                  for i in range(len(group))]
        timeseqs2.append(times2)

        # Calculate time differences since midnight
        midnight = group['timestamp'].iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
        times3 = [(t - midnight).seconds for t in group['timestamp']]
        timeseqs3.append(times3)

        # Get the day of the week
        times4 = [t.weekday() for t in group['timestamp']]
        timeseqs4.append(times4)

        # nb_itemseqs.append(group['related_offers'].apply(lambda x: len(x.split(':'))).tolist())

        # Calculate timeseqsF
        timeseqsF.append([times2[-1] - t for t in times2])

    # Create DataFrame
    data = {
        'caseid': caseids,
        'lines': lines,
        'timeseqs': timeseqs,
        'timeseqs2': timeseqs2,
        'timeseqs3': timeseqs3,
        'timeseqs4': timeseqs4,
        # 'nb_itemseqs': nb_itemseqs,
        'timeseqsF': timeseqsF
    }
    res = pd.DataFrame(data)

    return res


def prepare_character_encoding(lines):
    # Get all unique characters from the lines
    chars = list(set().union(*map(set, lines)))
    chars.sort()

    # Print the total number of characters
    print('Total characters: {}'.format(len(chars)))

    # Create dictionaries for character-to-index and index-to-character mappings
    char_indices = {char: i for i, char in enumerate(chars)}
    indices_char = {i: char for i, char in enumerate(chars)}
    char_act = {char: ord(char) - 64 for char in chars}

    return chars, char_indices, indices_char, char_act


def construct_sequences(preprocessed_data):
    sequences = []

    for index, row in tqdm(preprocessed_data.iterrows(), total=len(preprocessed_data), desc='Constructing sequences'):
        line = row['lines']
        line_t = row['timeseqs']
        line_t2 = row['timeseqs2']
        line_t3 = row['timeseqs3']
        line_t4 = row['timeseqs4']
        # line_t5 = row['nb_itemseqs']
        line_t6 = row['timeseqsF']

        for i in range(0, len(line)):
            if i == 0:
                continue

            sequence = {
                'sequence': line[0:i],
                'time_seq': line_t[0:i],
                'time_seq2': line_t2[0:i],
                'time_seq3': line_t3[0:i],
                'time_seq4': line_t4[0:i],
                # 'nb_itemseq': line_t5[0:i],
                'next_char': line[i],
                'next_char_time': 0 if i == len(line) - 1 else line_t[i],
                'next_char_timeF': 0 if i == len(line) - 1 else line_t6[i]
            }

            sequences.append(sequence)

    return pd.DataFrame(sequences)


def prepare_lstm_input_and_targets(sequences, maxlen, char_indices, char_act, scaler):
    lstm_input = []
    act_targets = []
    time_targets = []

    # Collect time sequence features for normalization
    time_features_all = []

    progress_bar = tqdm(total=len(sequences), desc='Preparing LSTM input and targets')
    for _, row in sequences.iterrows():
        line = row['sequence']
        time_seq = row['time_seq']
        time_seq2 = row['time_seq2']
        time_seq3 = row['time_seq3']
        time_seq4 = row['time_seq4']
        # nb_itemseq = row['nb_itemseq']
        next_char = row['next_char']
        next_char_time = row['next_char_time']
        next_char_timeF = row['next_char_timeF']

        l = len(line)
        pos_list = list(range(l))
        pos = [a + 1 for a in pos_list]
        # Initialize input vectors
        x_lstm = np.zeros((maxlen, 6), dtype=np.float32)
        left_pad = maxlen - l
        # Encode the sequence
        for t, char in enumerate(line):
            x_lstm[left_pad+t, 0] = char_act[char]
        x_lstm[left_pad:, 1] = pos
        # Fill the rest with time sequence features
        x_lstm[left_pad:, 2] = time_seq
        x_lstm[left_pad:, 3] = time_seq2
        x_lstm[left_pad:, 4] = time_seq3
        x_lstm[left_pad:, 5] = time_seq4
        # x_lstm[:l, 5] = nb_itemseq

        # Prepare target vectors
        y_act = np.zeros(len(char_indices), dtype=np.float32)
        y_act[char_indices[next_char]] = 1.0

        y_time = np.array([next_char_time, next_char_timeF], dtype=np.float32)

        lstm_input.append(x_lstm)
        act_targets.append(y_act)
        time_targets.append(y_time)

        progress_bar.update(1)

    progress_bar.close()

    # Fit scaler on the training data
    # time_features_all = np.concatenate([x_lstm[:, 1:5] for x_lstm in lstm_input])
    # scaler.fit(time_features_all)

    # Normalize time sequence features by dividing each feature by its mean
    time_features_mean = np.mean(np.array([x_lstm[:, -4:] for x_lstm in lstm_input]), axis=(0, 1))
    # Apply scaler to time sequence features
    for x_lstm in lstm_input:
        # x_lstm[:, 1:6] = scaler.transform(x_lstm[:, 1:6])
        x_lstm[:, -4:] /= time_features_mean
        x_lstm[:, -1] /= 7

    time_targets_array = np.array(time_targets)
    time_target_means = np.mean(time_targets_array, axis=0)
    # Normalize time targets by dividing each target by its mean
    normalized_time_targets = time_targets_array / time_target_means

    return np.array(lstm_input), np.array(act_targets), normalized_time_targets, time_target_means




def graph_train_test_split(data_df, graph_frac=0.1, train_frac=0.7, test_frac=0.2, random_state=None):
    case_list = []
    for case_id, group in data_df.groupby('CaseID'):
        group['CaseID'] = case_id
        case_list.append(group)
    # Shuffle the DataFrame
    case_list_shuffled = random.sample(case_list, k=len(case_list))

    # Calculate the number of rows for each part
    num_rows = len(case_list_shuffled)
    graph_size = int(num_rows * graph_frac)
    train_size = int(num_rows * train_frac)
    test_size = num_rows - graph_size - train_size

    # Split the data
    graph_data = case_list_shuffled[:graph_size]
    graph_data = pd.concat(graph_data, ignore_index=True)
    train_data = case_list_shuffled[graph_size:graph_size + train_size]
    train_data = pd.concat(train_data, ignore_index=True)
    test_data = case_list_shuffled[graph_size + train_size:]
    test_data = pd.concat(test_data, ignore_index=True)

    return graph_data, train_data, test_data
