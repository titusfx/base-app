import hashlib
import pickle
import os


def calculate_sha256(video_input):
    """
    Calculate the SHA-256 hash of the video file.
    """
    sha256_hash = hashlib.sha256()
    with open(video_input, "rb") as f:
        # Read and update hash in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_save_path(video_input, data_type="candidates", local_folder=None):
    """
    Generates a file path for saving or loading data (candidates/games) based on the video hash.
    """
    video_hash = calculate_sha256(video_input)  # Calculate the video hash
    # If local_folder is provided, check if it exists, if not create it
    if local_folder is not None:
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        return f"{local_folder}/{video_hash}_{data_type}.pkl"

    return f"{video_hash}_{data_type}.pkl"

    # Save data as pickle files


def save_data(video_input, data, data_type="candidates", local_folder=None):
    """
    Saves the given data (candidates/games) to a file.
    """
    file_path = get_save_path(video_input, data_type, local_folder)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"{data_type.capitalize()} saved to {file_path}")


def load_data(video_input, data_type="candidates", local_folder=None):
    """
    Loads the saved data (candidates/games) from a file, if it exists.
    """
    file_path = get_save_path(video_input, data_type, local_folder)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"{data_type.capitalize()} loaded from {file_path}")
        return data
    else:
        return None


def load_pickle_by_path(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"loaded from {file_path}")
        return data
    else:
        return None
