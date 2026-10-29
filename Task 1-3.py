# Deliverable 2: Campus Coffee Shop Sales Pipeline
# Task 1: File ingestion and regular expressions
# Task 2: Recursive utility functions
# Task 3: Functions for cleaning and analysing data using NumPy

# Makes type hints like str | Path work on Python 3.9.
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

DATA_DIR = Path('data')
# Epsilon from the assignment that prevents division by zero.
EPSILON = 1e-6


# I used named groups (?P<name>...) so that each field gets a name.
# Then match.groupdict() returns a ready-made dictionary instead
# of getting the fields one by one with number indexes.
LOG_LINE_PATTERN = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+"
    r"\[(?P<register_id>[A-Za-z0-9_]+)\]\s+"
    # Non-greedy (.+?) so item names with spaces also work.
    r"item=(?P<item>.+?)\s+"
    r"price=(?P<price>\d+(?:\.\d+)?)\s+"
    r"(?P<currency>[A-Z]{3})\s+"
    # Only the three valid statuses are accepted.
    # $ = end of the line, so lines with extra text after
    # the status are not counted as valid.
    r"\((?P<status>SUCCESS|REFUND|CANCELLED)\)$"
)


# %%
# Task 1: File ingestion and regular expressions


# Robust file reader
def read_log_file(file_path: str | Path) -> tuple[str, str]:
    """
    Read a text file that is encoded in either UTF-8 or ISO-8859-1.

    The file is read as raw bytes and decoded as UTF-8 first. If that
    fails it's decoded as ISO-8859-1 (Latin-1), so the encoding does
    not have to be hard-coded for each file.

    Parameters
    ----------
    file_path : str | Path
        Path to the file that should be read.

    Returns
    -------
    tuple[str, str]
        The text content of the file and the name of the encoding used.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    # Reading bytes lets the same content be decoded more than once.
    raw_bytes = Path(file_path).read_bytes()
    try:
        return raw_bytes.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        # Latin-1 can decode any byte, so it must be tried last. If it
        # came first, UTF-8 files would be read with wrong characters.
        return raw_bytes.decode("iso-8859-1"), "iso-8859-1"


# Log parser using regular expressions
def parse_log_text(log_text: str) -> list[dict[str, str | float]]:
    """
    Extract valid transaction records from the text of a log file.

    Lines that do not match the sales format, such as printer warnings
    and network messages, are ignored.

    Parameters
    ----------
    log_text : str
        The full text content of a log file.

    Returns
    -------
    list[dict[str, str | float]]
        One dictionary per valid transaction with the keys
        ``timestamp``, ``register_id``, ``item``, ``price``,
        ``currency`` and ``status``. ``price`` is a float, the other
        values are strings.
    """
    records: list[dict[str, str | float]] = []
    # splitlines() handles both \n (Mac/Linux) and \r\n (Windows),
    # so line breaks from different cash registers are treated equally.
    for line in log_text.splitlines():
        match = LOG_LINE_PATTERN.match(line.strip())

        # Noise lines are skipped because they do not match.
        if match is None:
            continue
        record: dict[str, str | float] = match.groupdict()

        # The price is converted to a float because the assignment
        # requires it, and because it is used in NumPy and pandas later.
        record["price"] = float(match.group("price"))
        records.append(record)

    return records


# Test of Task 1
# A small test string with both valid lines and noise, to check that
# the parser only extracts the sales lines.
example_text = (
    "--- PRINTER PAPER LOW ---\n"
    "[2026-10-15 08:30:00] [POS_01] item=Latte price=4.50 EUR (SUCCESS)\n"
    "=== NETWORK SYNC OK ===\n"
    "[DEBUG] BARCODE SCANNER REBOOTED\n"
    "[2026-10-15 08:32:10] [POS_01] item=Muffin price=999.00 NOK (REFUND)\n"
)

print("--- Task 1: File ingestion and regular expressions ---")
parsed = parse_log_text(example_text)
for transaction in parsed:
    print(transaction)

# Only the two sales lines should be kept, not the three noise lines.
assert len(parsed) == 2
assert parsed[0]["item"] == "Latte"
assert isinstance(parsed[0]["price"], float)
assert parsed[1]["status"] == "REFUND"


# %%
# Task 2: Recursive utility functions


# Recursive directory search
def find_log_files(directory: str | Path,
                   extension: str = ".log") -> list[Path]:
    """
    Recursively find all files with a given extension in a directory.

    The function goes through every item in the directory with
    ``Path.iterdir()``. Subdirectories are searched by calling the
    function again (recursion). ``glob``, ``rglob`` and ``os.walk`` are
    deliberately not used.

    Parameters
    ----------
    directory : str | Path
        The directory to start searching in.
    extension : str, optional
        File extension to look for, by default ".log".

    Returns
    -------
    list[Path]
        Paths to all matching files in the directory and all its
        subdirectories, sorted alphabetically within each folder.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    """
    # The assignment requires the input to be converted to a Path.
    # This lets the function accept both "data" (str) and Path("data").
    directory = Path(directory)
    found_files: list[Path] = []

    # sorted() gives the same order on every computer.
    for item in sorted(directory.iterdir()):
        if item.is_dir():
            # The recursion stops when a folder has no subfolders.
            found_files.extend(find_log_files(item, extension))
        elif item.is_file() and item.suffix == extension:
            found_files.append(item)

    return found_files


# Helper function for reading the config file
def load_store_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load the store configuration from a JSON file.

    Parameters
    ----------
    config_path : str | Path
        Path to ``store_config.json``.

    Returns
    -------
    dict[str, Any]
        The JSON content as nested Python dictionaries and lists.
    """
    with Path(config_path).open(encoding="utf-8") as file:
        return json.load(file)


# Recursive search for active register IDs
def find_active_registers(data: Any) -> set[str]:
    """
    Recursively collect the IDs of all active registers.

    Traverses any combination of nested dictionaries and lists. A
    dictionary counts as an active register if it has an ``"id"`` key
    with a string value and ``"active"`` set to ``True``.

    Parameters
    ----------
    data : Any
        Nested structure of dictionaries and lists, for example the
        content of ``store_config.json``.

    Returns
    -------
    set[str]
        The IDs of all registers where ``"active"`` is ``True``.
    """
    # I use a set because the assignment requires it, and because a set
    # removes duplicates if the same ID appears in multiple places.
    active_ids: set[str] = set()

    if isinstance(data, dict):
        # "is True" so that e.g. the number 1 does not count as active.
        register_id = data.get("id")
        if data.get("active") is True and isinstance(register_id, str):
            active_ids.add(register_id)

        # Lists and dictionaries can be nested inside the values.
        for value in data.values():
            # |= combines two sets (union).
            active_ids |= find_active_registers(value)

    elif isinstance(data, list):
        for element in data:
            active_ids |= find_active_registers(element)

    # Strings, numbers and booleans have no deeper levels, so the
    # recursion stops here (base case).
    return active_ids


# Test of Task 2
print("\n--- Task 2: Recursive utility functions ---")

# The example from the assignment, where only POS_01 is active.
example_config = {
    "company": "Campus Coffee",
    "regions": [
        {
            "region_name": "Main Campus",
            "stores": [
                {
                    "name": "Library Cafe",
                    "registers": [
                        {"id": "POS_01", "active": True},
                        {"id": "POS_02", "active": False},
                    ],
                }
            ],
        }
    ],
}
assert find_active_registers(example_config) == {"POS_01"}

# The real data from generate_data_for_d2.py. The data folder must be
# in the same folder as this file.
log_files = find_log_files(DATA_DIR)
print(f"Found {len(log_files)} log files:")
for log_file in log_files:
    print(" ", log_file)

store_config = load_store_config(DATA_DIR / "store_config.json")
active_registers = find_active_registers(store_config)
print("Active registers:", sorted(active_registers))

# 3 stores x 3 registers = 9 log files.
assert len(log_files) == 9
assert active_registers == {"POS_01", "POS_02", "POS_03"}


# %%
# Task 3: Functions for cleaning and analysing data using NumPy


# Remove price outliers
def remove_price_outliers(prices: npt.ArrayLike,
                          n_std: float = 3.0) -> np.ndarray:
    """
    Replace extreme price outliers with the median of the valid prices.

    A value is treated as an outlier if it deviates from the mean by
    more than ``n_std`` standard deviations, i.e.
    ``abs(x - mean) > n_std * std``.

    Parameters
    ----------
    prices : npt.ArrayLike
        Price data, e.g. a list, a pandas Series or a NumPy array.
    n_std : float, optional
        Number of standard deviations used as the limit, by default 3.0.

    Returns
    -------
    np.ndarray
        A cleaned 1D float array of the same length as the input.
    """
    # np.array makes a copy, so the original data is not changed.
    cleaned = np.array(prices, dtype=float).ravel()

    # mean() of an empty array gives a warning, so return early.
    if cleaned.size == 0:
        return cleaned

    mean = cleaned.mean()
    std = cleaned.std()

    # True where the value is an outlier. Calculated for the whole array
    # at once (vectorised) instead of with a for loop.
    is_outlier = np.abs(cleaned - mean) > n_std * std

    # The median uses only the valid prices, so the 999 values do not
    # pull it up. The median is less affected by extremes than the mean.
    if is_outlier.any():
        valid_median = np.median(cleaned[~is_outlier])
        cleaned[is_outlier] = valid_median

    return cleaned


# Compute refund ratio
def compute_refund_ratio(sales_count: np.ndarray,
                         refund_count: np.ndarray) -> np.ndarray:
    """
    Compute the element-wise refund ratio.

    Uses ``refund_count / (sales_count + 1e-6)``. The small epsilon
    prevents division by zero when an item has no sales.

    Parameters
    ----------
    sales_count : np.ndarray
        Number of sales per item.
    refund_count : np.ndarray
        Number of refunds per item, in the same order as
        ``sales_count``.

    Returns
    -------
    np.ndarray
        Refund ratio for each element.
    """
    # float, so lists and pandas Series also work as input.
    sales = np.asarray(sales_count, dtype=float)
    refunds = np.asarray(refund_count, dtype=float)

    # NumPy divides element by element.
    return refunds / (sales + EPSILON)


# Test of Task 3
print("\n--- Task 3: Cleaning and analysing data using NumPy ---")

# The 3-sigma rule needs enough data points. With only a few prices, one
# outlier pulls up both the mean and the std so much that it is missed.
test_prices = [3.0, 4.5, 4.2, 3.5, 3.8, 7.5, 3.1, 4.4, 4.0, 3.6,
               3.9, 7.2, 2.9, 4.6, 4.3, 3.4, 3.7, 7.8, 3.2, 4.1, 999.0]
cleaned_test = remove_price_outliers(test_prices)
print("Price before:", test_prices[-1], "-> after:", cleaned_test[-1])
assert 999.0 not in cleaned_test
assert len(cleaned_test) == len(test_prices)

# All the real prices from the log files (uses Task 1 and 2).
all_records = []
for log_file in log_files:
    text, _ = read_log_file(log_file)
    all_records.extend(parse_log_text(text))

real_prices = np.array([record["price"] for record in all_records])
cleaned_real = remove_price_outliers(real_prices)
print("Number of prices:", real_prices.size)
print("999 glitches before cleaning:", int(np.sum(real_prices == 999.0)))
print("999 glitches after cleaning:", int(np.sum(cleaned_real == 999.0)))
assert not np.any(cleaned_real == 999.0)

# The last item has no sales, to check that there is no zero division.
ratios = compute_refund_ratio(np.array([10, 4, 0]), np.array([2, 1, 0]))
print("Refund ratios:", np.round(ratios, 3))
assert np.allclose(ratios, [0.2, 0.25, 0.0])
