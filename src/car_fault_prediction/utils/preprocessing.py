import logging

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


logger = logging.getLogger(__name__)


def fill_missing(data, strategy_numeric="auto", save_indicators=False):
    """Fill missing values while preserving the original preprocessing rules."""
    for column_name in data.columns:
        if data[column_name].isnull().sum() == 0:
            continue

        try:
            if save_indicators:
                data[f"is_missing_{column_name}"] = data[column_name].isnull().astype(int)

            if data[column_name].dtype in ["float64", "int64", "bool"]:
                if strategy_numeric == "auto":
                    skew_value = data[column_name].skew()
                    if pd.notna(skew_value) and skew_value > 1:
                        data[column_name] = data[column_name].fillna(data[column_name].median())
                    else:
                        data[column_name] = data[column_name].fillna(data[column_name].mean())
                elif strategy_numeric == "median":
                    data[column_name] = data[column_name].fillna(data[column_name].median())
                elif strategy_numeric == "mean":
                    data[column_name] = data[column_name].fillna(data[column_name].mean())
                else:
                    raise ValueError("strategy_numeric must be 'mean', 'median', or 'auto'")
            else:
                mode_value = data[column_name].mode()
                if not mode_value.empty:
                    data[column_name] = data[column_name].fillna(mode_value[0])
                else:
                    data[column_name] = data[column_name].fillna("Unknown")
        except Exception as error:
            logger.warning("Error while filling missing values for %s: %s", column_name, error)
            continue

    return data


def encode_categorical_columns(data, encoders_path=None):
    """Encode categorical columns using the same training-time strategy."""
    encoded_data = data.copy()
    label_encoders = {}

    try:
        if encoders_path:
            label_encoders = joblib.load(encoders_path)
            binary_columns = [
                column_name
                for column_name in label_encoders.keys()
                if column_name not in {"onehot_encoder", "onehot_columns"}
            ]
            multi_value_columns = label_encoders.get("onehot_columns", [])
            onehot_encoder = label_encoders.get("onehot_encoder")
        else:
            categorical_columns = encoded_data.select_dtypes(include=["object"]).columns
            binary_columns = []
            multi_value_columns = []

            for column_name in categorical_columns:
                unique_values = encoded_data[column_name].nunique()
                if unique_values == 2:
                    binary_columns.append(column_name)
                elif unique_values > 2:
                    multi_value_columns.append(column_name)

            onehot_encoder = None

        for column_name in binary_columns:
            if encoders_path and column_name in label_encoders:
                label_encoder = label_encoders[column_name]
                try:
                    encoded_data[column_name] = label_encoder.transform(encoded_data[column_name])
                except ValueError:
                    logger.warning(
                        "Column %s contains unseen values. Falling back to the first known class.",
                        column_name,
                    )
                    encoded_data[column_name] = encoded_data[column_name].map(
                        lambda value: value if value in label_encoder.classes_ else label_encoder.classes_[0]
                    )
                    encoded_data[column_name] = label_encoder.transform(encoded_data[column_name])
            else:
                label_encoder = LabelEncoder()
                encoded_data[column_name] = label_encoder.fit_transform(encoded_data[column_name])
                label_encoders[column_name] = label_encoder

        if multi_value_columns:
            if encoders_path and onehot_encoder is not None:
                try:
                    onehot_encoded = onehot_encoder.transform(encoded_data[multi_value_columns])
                    onehot_columns = onehot_encoder.get_feature_names_out(multi_value_columns)
                except ValueError:
                    logger.warning(
                        "Columns %s contain unseen values. Falling back to known encoder categories.",
                        multi_value_columns,
                    )
                    for column_name in multi_value_columns:
                        category_index = multi_value_columns.index(column_name)
                        encoded_data[column_name] = encoded_data[column_name].map(
                            lambda value: (
                                value
                                if value in onehot_encoder.categories_[category_index]
                                else onehot_encoder.categories_[category_index][0]
                            )
                        )
                    onehot_encoded = onehot_encoder.transform(encoded_data[multi_value_columns])
                    onehot_columns = onehot_encoder.get_feature_names_out(multi_value_columns)
            else:
                onehot_encoder = OneHotEncoder(sparse_output=False, drop="first")
                onehot_encoded = onehot_encoder.fit_transform(encoded_data[multi_value_columns])
                onehot_columns = onehot_encoder.get_feature_names_out(multi_value_columns)
                label_encoders["onehot_encoder"] = onehot_encoder
                label_encoders["onehot_columns"] = multi_value_columns

            onehot_frame = pd.DataFrame(
                onehot_encoded,
                columns=onehot_columns,
                index=encoded_data.index,
            )
            encoded_data = encoded_data.drop(columns=multi_value_columns)
            encoded_data = pd.concat([encoded_data, onehot_frame], axis=1)

        if not encoders_path:
            joblib.dump(label_encoders, "encoders.pkl")

        return encoded_data, label_encoders
    except Exception as error:
        logger.warning("Categorical column encoding error: %s", error)
        return encoded_data, label_encoders
