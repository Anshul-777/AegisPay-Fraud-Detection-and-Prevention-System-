import pandas as pd
import numpy as np
import joblib
import os
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
import collections # For optimized rolling features

# --- Global variables to store loaded artifacts ---
onehot_encoder = None
location_coords = {}
sender_most_frequent_coords = {}
X_train_columns = []

# --- Paths to artifacts ---
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'onehot_encoder.joblib')
LOCATION_COORDS_PATH = os.path.join(ARTIFACTS_DIR, 'location_coords.joblib')
SENDER_MOST_FREQUENT_COORDS_PATH = os.path.join(ARTIFACTS_DIR, 'sender_most_frequent_coords.joblib')
X_TRAIN_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, 'X_train_columns.joblib')

def load_artifacts():
    """Loads all necessary artifacts for the feature engineering pipeline."""
    global onehot_encoder, location_coords, sender_most_frequent_coords, X_train_columns
    try:
        onehot_encoder = joblib.load(ENCODER_PATH)
        # print(f"Loaded OneHotEncoder from {ENCODER_PATH}")

        location_coords = joblib.load(LOCATION_COORDS_PATH)
        # print(f"Loaded location_coords from {LOCATION_COORDS_PATH}")

        sender_most_frequent_coords = joblib.load(SENDER_MOST_FREQUENT_COORDS_PATH)
        # print(f"Loaded sender_most_frequent_coords from {SENDER_MOST_FREQUENT_COORDS_PATH}")

        X_train_columns = joblib.load(X_TRAIN_COLUMNS_PATH)
        # print(f"Loaded X_train_columns from {X_TRAIN_COLUMNS_PATH}")

    except FileNotFoundError as e:
        print(f"Error loading artifact: {e}. Please ensure all artifacts are in the correct directory relative to the script.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred while loading artifacts: {e}")
        raise e

# --- Helper Functions ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth using Haversine formula."""
    R = 6371  # Radius of Earth in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def get_coordinates(city, country):
    """Retrieves synthetic coordinates for a city-country pair from loaded artifacts."""
    key = (city, country)
    return location_coords.get(key, (np.nan, np.nan)) # Return NaN if not found

def get_most_frequent_sender_coords(sender_id):
    """Retrieves most frequent location coordinates for a sender from loaded artifacts."""
    return sender_most_frequent_coords.get(sender_id, (np.nan, np.nan)) # Return NaN if not found

# --- Feature Generation Functions ---
def generate_geospatial_coords(df):
    """Generates sender and receiver latitude/longitude using loaded artifacts."""
    df['sender_latitude'] = df.apply(lambda row: get_coordinates(row['sender_city'], row['sender_country'])[0], axis=1)
    df['sender_longitude'] = df.apply(lambda row: get_coordinates(row['sender_city'], row['sender_country'])[1], axis=1)
    df['receiver_latitude'] = df.apply(lambda row: get_coordinates(row['receiver_city'], row['receiver_country'])[0], axis=1)
    df['receiver_longitude'] = df.apply(lambda row: get_coordinates(row['receiver_city'], row['receiver_country'])[1], axis=1)
    return df

def generate_location_based_features(df):
    """Calculates sender-receiver distance, sender unusual location distance, and location change velocity."""
    df['sender_receiver_distance'] = df.apply(lambda row:
        haversine(row['sender_latitude'], row['sender_longitude'],
                  row['receiver_latitude'], row['receiver_longitude'])
        if pd.notnull(row['sender_latitude']) and pd.notnull(row['receiver_latitude']) else 0,
        axis=1
    )

    df['sender_most_freq_latitude'] = df['sender_id'].apply(lambda x: get_most_frequent_sender_coords(x)[0])
    df['sender_most_freq_longitude'] = df['sender_id'].apply(lambda x: get_most_frequent_sender_coords(x)[1])

    df['sender_unusual_location_distance'] = df.apply(lambda row:
        haversine(row['sender_latitude'], row['sender_longitude'],
                  row['sender_most_freq_latitude'], row['sender_most_freq_longitude'])
        if pd.notnull(row['sender_most_freq_latitude']) and pd.notnull(row['sender_latitude']) else 0,
        axis=1
    )

    df.drop(columns=['sender_most_freq_latitude', 'sender_most_freq_longitude'], errors='ignore', inplace=True)

    # For location_change_velocity, this needs historical data. For real-time inference on single records, it defaults to 0.
    # If processing a batch, we can calculate within the batch for existing users.
    df = df.sort_values(by=['sender_id', 'transaction_created_at'])
    df['time_diff_seconds'] = df.groupby('sender_id')['transaction_created_at'].diff().dt.total_seconds()
    df['prev_sender_latitude'] = df.groupby('sender_id')['sender_latitude'].shift(1)
    df['prev_sender_longitude'] = df.groupby('sender_id')['sender_longitude'].shift(1)

    df['distance_diff_km'] = df.apply(lambda row:
        haversine(row['sender_latitude'], row['sender_longitude'],
                  row['prev_sender_latitude'], row['prev_sender_longitude'])
        if pd.notnull(row['prev_sender_latitude']) else 0,
        axis=1
    )

    df['location_change_velocity'] = df.apply(lambda row:
        row['distance_diff_km'] / row['time_diff_seconds'] if row['time_diff_seconds'] > 0 else 0,
        axis=1
    )
    df['location_change_velocity'] = df['location_change_velocity'].fillna(0)

    df.drop(columns=['time_diff_seconds', 'prev_sender_latitude', 'prev_sender_longitude', 'distance_diff_km'], errors='ignore', inplace=True)
    return df

def generate_time_based_features(df):
    """Extracts temporal features and calculates user transaction frequencies."""
    df['transaction_hour'] = df['transaction_created_at'].dt.hour
    df['transaction_day_of_week'] = df['transaction_created_at'].dt.dayofweek
    df['transaction_day_of_month'] = df['transaction_created_at'].dt.day
    df['transaction_month'] = df['transaction_created_at'].dt.month

    df = df.sort_values(by=['sender_id', 'transaction_created_at'])
    df['time_since_last_transaction_user'] = df.groupby('sender_id')['transaction_created_at'].diff().dt.total_seconds().fillna(0)

    # For real-time inference, these rolling counts are hard to get without a stateful service.
    # For a batch, we can compute within the batch for a specific user.
    def calculate_time_window_features_batch(group_df):
        group_df = group_df.sort_values(by='transaction_created_at')
        results = []
        for i in range(len(group_df)):
            current_row = group_df.iloc[i]
            current_time = current_row['transaction_created_at']
            previous_transactions = group_df.iloc[:i]

            count_1h = ((previous_transactions['transaction_created_at'] > (current_time - timedelta(hours=1))) &
                        (previous_transactions['transaction_created_at'] < current_time)).sum()
            count_24h = ((previous_transactions['transaction_created_at'] > (current_time - timedelta(hours=24))) &
                         (previous_transactions['transaction_created_at'] < current_time)).sum()
            count_7d = ((previous_transactions['transaction_created_at'] > (current_time - timedelta(days=7))) &
                        (previous_transactions['transaction_created_at'] < current_time)).sum()
            results.append({
                'transactions_in_last_hour_user': count_1h,
                'transactions_in_last_24h_user': count_24h,
                'transactions_in_last_7d_user': count_7d
            })
        return pd.DataFrame(results, index=group_df.index)

    temp_time_features = df.groupby('sender_id', group_keys=False).apply(calculate_time_window_features_batch)
    df = df.merge(temp_time_features, left_index=True, right_index=True, how='left')

    df['transactions_in_last_hour_user'] = df['transactions_in_last_hour_user'].fillna(0)
    df['transactions_in_last_24h_user'] = df['transactions_in_last_24h_user'].fillna(0)
    df['transactions_in_last_7d_user'] = df['transactions_in_last_7d_user'].fillna(0)

    return df

def generate_user_based_aggregated_features(df):
    """Calculates user's historical transaction aggregates for amount and unique receivers."""
    df = df.sort_values(by=['sender_id', 'transaction_created_at'])

    def calculate_user_rolling_features_batch(group):
        group_sorted = group.sort_values('transaction_created_at')
        group_temp_indexed = group_sorted.set_index('transaction_created_at')

        avg_amount_roll = group_temp_indexed['amount'].rolling(window='30D', closed='left').mean()
        max_amount_roll = group_temp_indexed['amount'].rolling(window='30D', closed='left').max()
        trans_count_roll = group_temp_indexed['transaction_id'].rolling(window='30D', closed='left').count()
        unique_receivers_roll = group_temp_indexed['receiver_id'].rolling(window='30D', closed='left').apply(lambda y: y.nunique(), raw=False)

        result_df = pd.DataFrame(index=group_sorted.index)
        result_df['user_avg_amount_past_30d'] = avg_amount_roll.reset_index(drop=True).values
        result_df['user_max_amount_past_30d'] = max_amount_roll.reset_index(drop=True).values
        result_df['user_transaction_count_past_30d'] = trans_count_roll.reset_index(drop=True).values
        result_df['user_unique_receivers_past_30d'] = unique_receivers_roll.reset_index(drop=True).values
        return result_df

    temp_user_features = df.groupby('sender_id', group_keys=False).apply(calculate_user_rolling_features_batch)
    df = df.merge(temp_user_features, left_index=True, right_index=True, how='left')

    df['user_avg_amount_past_30d'] = df['user_avg_amount_past_30d'].fillna(0)
    df['user_max_amount_past_30d'] = df['user_max_amount_past_30d'].fillna(0)
    df['user_transaction_count_past_30d'] = df['user_transaction_count_past_30d'].fillna(0)
    df['user_unique_receivers_past_30d'] = df['user_unique_receivers_past_30d'].fillna(0)

    df['balance_change_percentage'] = ((df['balance_before'] - df['balance_after']) / df['balance_before']).fillna(0)
    df.loc[df['balance_before'] == 0, 'balance_change_percentage'] = 0
    return df

def generate_amount_based_features(df):
    """Creates amount_relative_to_user_avg feature."""
    df['amount_relative_to_user_avg'] = df.apply(lambda row:
        row['amount'] / row['user_avg_amount_past_30d']
        if row['user_avg_amount_past_30d'] != 0 else 0,
        axis=1
    )
    return df

def generate_device_based_features(df):
    """Generates features related to device usage."""
    df = df.sort_values(by=['sender_id', 'transaction_created_at'])

    def is_new_device_batch(group):
        new_device_flags = []
        seen_devices = set()
        for _, row in group.iterrows():
            if row['transaction_device_id'] not in seen_devices:
                new_device_flags.append(1)
                seen_devices.add(row['transaction_device_id'])
            else:
                new_device_flags.append(0)
        return pd.Series(new_device_flags, index=group.index)

    df['is_new_device_for_user'] = df.groupby('sender_id', group_keys=False).apply(is_new_device_batch)
    df['is_new_device_for_user'] = df['is_new_device_for_user'].fillna(0)

    def count_unique_devices_rolling_optimized_batch(group):
        group_sorted = group.sort_values('transaction_created_at')
        results = pd.Series(index=group_sorted.index, dtype=float)
        window_devices = collections.deque()
        device_counts = collections.Counter()

        for i, (idx, row) in enumerate(group_sorted.iterrows()):
            current_time = row['transaction_created_at']
            current_device_id = row['transaction_device_id']
            window_start_time = current_time - timedelta(days=30)

            while window_devices and window_devices[0][0] < window_start_time:
                old_timestamp, old_device_id = window_devices.popleft()
                device_counts[old_device_id] -= 1
                if device_counts[old_device_id] == 0:
                    del device_counts[old_device_id]

            results.loc[idx] = len(device_counts)

            window_devices.append((current_time, current_device_id))
            device_counts[current_device_id] += 1
        return results

    df['num_unique_devices_user_past_30d'] = df.groupby('sender_id', group_keys=False).apply(count_unique_devices_rolling_optimized_batch)
    df['num_unique_devices_user_past_30d'] = df['num_unique_devices_user_past_30d'].fillna(0)
    return df

def apply_one_hot_encoding(df_to_encode, encoder):
    """Applies one-hot encoding to specified categorical columns using a pre-fitted encoder."""
    categorical_cols_to_encode = ['status'] # Hardcode based on training

    # Transform the categorical column using the loaded encoder
    encoded_features = encoder.transform(df_to_encode[categorical_cols_to_encode])

    # Create a DataFrame from the encoded features with appropriate column names
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols_to_encode)
    df_encoded_part = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_to_encode.index)

    # Concatenate the new encoded features with the original DataFrame (excluding the original categorical column)
    df_processed = pd.concat([df_to_encode.drop(columns=categorical_cols_to_encode, errors='ignore'), df_encoded_part], axis=1)
    return df_processed

def preprocess_new_data(raw_data_df):
    """Orchestrates all feature engineering steps for new raw transaction data.

    Args:
        raw_data_df (pd.DataFrame): A DataFrame containing new raw transaction data.
                                   Must include columns: 'sender_id', 'receiver_id', 'transaction_id',
                                   'amount', 'description', 'payment_method', 'status', 'transaction_created_at',
                                   'transaction_payment_time', 'balance_before', 'balance_after',
                                   'sender_country', 'sender_state', 'sender_city',
                                   'receiver_country', 'receiver_state', 'receiver_city',
                                   'transaction_device_id', 'device_platform', 'device_browser',
                                   'device_user_agent', 'device_owner_user_id'

    Returns:
        pd.DataFrame: A DataFrame with engineered features, ready for model prediction.
                      Columns will match X_train_columns exactly.
    """
    global onehot_encoder, location_coords, sender_most_frequent_coords, X_train_columns
    if onehot_encoder is None or not location_coords or not sender_most_frequent_coords or not X_train_columns:
        load_artifacts()

    # Ensure transaction_created_at is datetime
    processed_df = raw_data_df.copy()
    processed_df['transaction_created_at'] = pd.to_datetime(processed_df['transaction_created_at'])

    # Apply feature generation functions sequentially
    processed_df = generate_geospatial_coords(processed_df)
    processed_df = generate_location_based_features(processed_df)
    processed_df = generate_time_based_features(processed_df)
    processed_df = generate_user_based_aggregated_features(processed_df)
    processed_df = generate_amount_based_features(processed_df)
    processed_df = generate_device_based_features(processed_df)

    # Pruning (replicate advanced pruning steps from training)
    columns_to_drop = [
        'transaction_id', 'description', 'device_user_agent',
        'sender_name', 'sender_email', 'sender_dob', 'sender_upi_id',
        'receiver_name', 'receiver_upi',
        'sender_bank_holder', 'sender_bank_name', 'sender_account_no', 'sender_ifsc',
        'receiver_bank_holder', 'receiver_bank_name', 'receiver_account_no', 'receiver_ifsc',
        'sender_country', 'sender_state', 'sender_city',
        'receiver_country', 'receiver_state', 'receiver_city',
        'transaction_created_at', 'transaction_payment_time',
        'device_created_at', 'device_last_used_at',
        'transaction_device_id', 'device_owner_user_id',
        'sender_id', 'receiver_id', 'sender_phone', 'sender_initial_balance'
    ]
    processed_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Apply One-Hot Encoding
    processed_df = apply_one_hot_encoding(processed_df, onehot_encoder)

    # Ensure column order and handle missing columns (for unseen categories during training)
    final_features_df = pd.DataFrame(columns=X_train_columns, index=processed_df.index)
    for col in X_train_columns:
        if col in processed_df.columns:
            final_features_df[col] = processed_df[col]
        else:
            final_features_df[col] = 0  # Fill with 0 for features not present in the new data

    return final_features_df

# Call load_artifacts once when the module is imported
load_artifacts()
