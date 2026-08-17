video_input = "/home/titusfx/Downloads/padel-data/6-games-26min-short.mp4"

from scoreboard_padel.basic_example.logic_basic_example import find_scoreboard

# For skipping gradrio
find_scoreboard(
    video_input,
    jump=2,
    in_seconds=True,
    desc=False,
)
