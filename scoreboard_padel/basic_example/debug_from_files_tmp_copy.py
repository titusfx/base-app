# from scoreboard_padel.basic_example.logic_basic_example import find_scoreboard
from scoreboard_padel.basic_example.utils_files import load_data, save_data
from scoreboard_padel.basic_example.game_utils import GameDetection, get_games
from scoreboard_padel.basic_example.video_utils import separate_game_videos


# For loading from file:
video_input = "/home/titusfx/Downloads/padel-data/6-games-26min-short.mp4"
local_folder = "./tmp_copy"
# Check if we have saved candidates and games for this video
candidates_points = load_data(
    video_input, data_type="candidates", local_folder=local_folder
)

desc = False
# get_games(candidates_points, False)
game_analyzer = GameDetection(desc=desc)
games = game_analyzer.analyze_candidates(candidates_points)
save_data(video_input, games, data_type="games", local_folder=local_folder)
separate_game_videos(video_input, games, desc, local_folder)

# games = load_data(video_input, data_type="games", local_folder=local_folder)


# # For skipping gradrio
# find_scoreboard(
#     "/home/titusfx/Downloads/padel-data/6-games-26min-short.mp4",
#     jump=5,
#     in_seconds=True,
#     desc=False,
# )
