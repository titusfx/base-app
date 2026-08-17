from utils_files import load_pickle_by_path

files = [
    "tmp/97c99cafebcb8ffe18e00f21670c9be480a5eeb139fd1dd9d45cc9aac86da3ff_candidates.pkl",
    "tmp/97c99cafebcb8ffe18e00f21670c9be480a5eeb139fd1dd9d45cc9aac86da3ff_games.pkl",
]
result = []
for file_path in files:
    data = load_pickle_by_path(file_path)
    result.append(data)

candidates = result[0]
games = result[1]

i = 1
fps = 27  # 54/2 frame_index/seconds
# The game i the last moment
games[i][-1]
# The game i-1 the first moment
games[i - 1][0]
# in desc

print(result)
