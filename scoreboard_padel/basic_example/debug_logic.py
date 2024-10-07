from scoreboard_padel.basic_example.logic_basic_example import find_scoreboard
from scoreboard_padel.basic_example.utils_files import load_data
from scoreboard_padel.basic_example.logic_basic_example import get_games


# For loading from file:
video_input = "/home/titusfx/Downloads/padel-data/6-games-26min-short.mp4"
local_folder = "./tmp"
# Check if we have saved candidates and games for this video
candidates_points = load_data(
    video_input, data_type="candidates", local_folder=local_folder
)

get_games(candidates_points, False)

games = load_data(video_input, data_type="games", local_folder=local_folder)


# # For skipping gradrio
# find_scoreboard(
#     "/home/titusfx/Downloads/padel-data/6-games-26min-short.mp4",
#     jump=5,
#     in_seconds=True,
#     desc=False,
# )
