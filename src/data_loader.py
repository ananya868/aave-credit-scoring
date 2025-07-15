import pandas as pd
import os
import sys
import json
from typing import Literal, Dict, Any, Union


"""Logging""" 
# Define a specific type for the log levels for clarity and safety.
LogLevel = Literal["INFO", "SUCCESS", "ERROR"]
# Module-level variable to control logging. True by default.
_logging_active: bool = True

def set_logging(enabled: bool) -> None:
    """
    Enables or disables logging for the module.

    By default, logging is enabled. Call this function with False at the
    start of your script to run in silent mode.

    :param enabled: Set to True to turn logging on, False to turn it off.
    """
    global _logging_active
    _logging_active = enabled
    if not enabled:
        print("Logging has been disabled.")


"""Helper and Utility Methods"""
def log_message(message: str, level: LogLevel = "INFO") -> None:
    """
    Prints a formatted, styled message to the console if logging is active.

    :param message: The string message to print.
    :param level: The type of message ('INFO', 'SUCCESS', 'ERROR'). Determines the styling.
    """
    # <<< This is the key change: check the state variable before doing anything.
    if not _logging_active:
        return

    if level == "INFO":
        print(f"\n\033[94m+---- [INFO] --------------------------------\033[0m")
        print(f"\033[94m| {message}\033[0m")
        print(f"\033[94m+------------------------------------------\033[0m")
    elif level == "SUCCESS":
        print(f"\n\033[92m+---- [SUCCESS] -----------------------------\033[0m")
        print(f"\033[92m| {message}\033[0m")
        print(f"\033[92m+------------------------------------------\033[0m")
    elif level == "ERROR":
        # Error messages should always print unless logging is explicitly disabled.
        print(f"\n\033[91m+---- [CRITICAL ERROR] ----------------------\033[0m")
        print(f"\033[91m| {message}\033[0m")
        print(f"\033[91m+------------------------------------------\033[0m")

def _validate_filepath(filepath: str) -> None:
    """
    Validates the existence and extension of the input file.
    Internal helper function.
    :param filepath: The path to the input data file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filepath}' was not found.")
    if not filepath.endswith('.json'):
        raise ValueError(f"The file '{filepath}' is not a JSON file.")

def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validates the basic structure and content of the loaded DataFrame.
    Internal helper function.
    :param df: The pandas DataFrame to validate.
    """
    if df.empty:
        raise ValueError("The loaded DataFrame is empty. No data to process.")
    required_columns = ['userWallet', 'action', 'timestamp', 'actionData']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The data is missing the following required columns: {missing_columns}")


"""Main Method"""
def load_transactions(filepath: str) -> pd.DataFrame:
    """
    Loads, validates, and cleans the transaction data from a JSON file.

    This is the main function of the module, orchestrating the loading and
    validation process. It terminates the program on critical failure.

    :param filepath: The path to the input JSON file.
    :return: A pandas DataFrame containing the transaction data.
    """
    log_message(f"Initiating data loading from: {filepath}")
    
    try:
        _validate_filepath(filepath)
        log_message("File path and extension are valid.")

        df = pd.read_json(filepath)
        log_message(f"Successfully loaded {len(df)} raw transaction records.")

        _validate_dataframe(df)
        log_message("DataFrame structure is valid. Essential columns are present.")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        def get_usd_value(data: Union[Dict[str, Any], None]) -> float:
            if not isinstance(data, dict):
                return 0.0
            amount_str = data.get('amount', '0')
            price_str = data.get('assetPriceUSD', '0')
            amount = float(amount_str) if amount_str is not None else 0.0
            price = float(price_str) if price_str is not None else 0.0
            return amount * price

        df['amountUSD'] = df['actionData'].apply(get_usd_value)
        log_message("Performed initial data cleaning and feature extraction (amountUSD).")

        log_message("Data loading and preparation complete.", level="SUCCESS")
        return df

    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        log_message(f"A critical error occurred: {e}", level="ERROR")
        sys.exit(1)
    except Exception as e:
        log_message(f"An unexpected error occurred during data loading: {e}", level="ERROR")
        sys.exit(1)


