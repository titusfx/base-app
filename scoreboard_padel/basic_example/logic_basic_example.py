# This define logic (greet method) and UI in the same file, probably is better in separated files for complex UI

# LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC  LOGIC
# Define a simple function that will be the core of your Gradio app
# import gradio as gr
import dis
import cv2
import os
import datetime
from collections import defaultdict


from scoreboard_padel.basic_example.reader import IReader, PaddleReader
from utils_files import save_data, load_data


# Initialize OCR
reader: IReader = PaddleReader("en")


# Ensure the output directory exists
output_dir = "output_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fps = -1


def get_frames(video_input, jump=1, in_seconds=True, desc=False):
    """
    A generator function to iterate through video frames either in normal order or reverse,
    based on the specified jump value in either seconds or frame indices.

    :param video_input: Path to the video file
    :param jump: Number of frames or seconds to skip between iterations (default is 1)
    :param in_seconds: If True, the `jump` value is treated as seconds; if False, it's treated as frames (default is True)
    :param desc: If True, iterate through the video in reverse order; if False, iterate in normal order (default is False)
    :yield: The current frame from the video and its frame index
    """
    global fps  # Use global fps variable to modify it

    # Open the video file
    cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_input}")

    # Get video properties
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate jump in frames
    if in_seconds:
        frame_jump = jump * fps
    else:
        frame_jump = jump

    # If descending, start from the last frame; otherwise, start from the first
    if desc:
        start_frame = frame_count - 1
        step = -frame_jump
    else:
        start_frame = 0
        step = frame_jump

    # Set initial frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Iterating over frames
    current_frame_index = start_frame
    while 0 <= current_frame_index < frame_count:
        # Capture the current frame
        ret, frame = cap.read()
        if not ret:
            break
        print(
            f"Frame-index: {current_frame_index}, timestamp {str(datetime.timedelta(seconds=int(current_frame_index / fps)))}, "
        )
        # Yield the frame and the current frame index
        yield current_frame_index, frame

        # Move to the next frame based on the step
        current_frame_index += step
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)

    cap.release()


def draw_bboxes_in_image(frame, results):
    """
    Save images with bounding boxes drawn around detected text.

    :param frame: The original image frame
    :param results: The results from easyocr readtext method
    :param output_folder: The folder where images will be saved
    """
    result = frame.copy()
    # Loop through each detected text result
    for i, (bbox, text, confidence) in enumerate(results):
        # Create a copy of the frame to draw on
        result = draw_frame_with_bbox(result, bbox, f"{text}-{confidence:.2f}")
        print(f"{i}. Image saved.")  # Feedback for each saved image
    return result


def save_frame(frame, output_path=None, extra={}):
    # Save the frame with the bounding box as an image
    if output_path is None:
        # convert extra data to filename
        filename = "_".join([f"{k}_{v}" for v, k in extra.items()])
        output_path = os.path.join(
            output_dir, f"frame_{filename}_{datetime.datetime.now()}.jpg"
        )

    cv2.imwrite(output_path, frame)

    return output_path


# Draw bounding box and save frame
def draw_frame_with_bbox(frame, bbox, text):
    # Ensure coordinates are integers
    x_min, y_min = map(int, bbox[0])
    x_max, y_max = map(int, bbox[2])

    # Create a copy of the frame to avoid modifying the original
    frame_copy = frame.copy()
    # Draw rectangle around the detected score
    cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(
        frame_copy,
        text,
        (x_min, y_min - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return frame_copy


# Helper function to check if two bounding boxes are in the same area
def match_bboxes(bbox1, bbox2, threshold=0.5):
    """
    Checks if two bounding boxes overlap significantly based on a threshold.
    :param bbox1: First bounding box coordinates.
    :param bbox2: Second bounding box coordinates.
    :param threshold: Minimum overlap ratio required to match boxes.
    :return: Boolean indicating whether the boxes match.
    """
    x1_min, y1_min = bbox1[0]
    x1_max, y1_max = bbox1[2]
    x2_min, y2_min = bbox2[0]
    x2_max, y2_max = bbox2[2]

    # Calculate intersection area
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection_area = x_overlap * y_overlap

    # Calculate areas of the bounding boxes
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute the overlap ratio
    overlap_ratio = intersection_area / min(area1, area2)

    return overlap_ratio >= threshold


# Detect the scoreboard area in the last frame based on the '40'
def organize_detections(
    detections,
    frame_index,
    jump,
    candidates,
    threshold=0.75,
    transcription_list=[],
    transcription={},
    target_scores=["40", "30", "15", "0"],
):
    discarded_detections = []
    for detection in detections:
        bbox, text, confidence = detection
        # If we detect the score we're looking for
        if confidence < threshold:
            # print(f"The text: {text} was discarded with confidence: {confidence}")
            discarded_detections.append(detection)
            continue
        object_data = {
            "bbox": bbox,
            "text": text,
            "confidence": confidence,
        }
        if text in target_scores:
            candidates.setdefault(frame_index, []).append(
                {
                    "frame_index": frame_index,
                    "timestamp": str(
                        datetime.timedelta(seconds=int(frame_index / fps))
                    ),
                    "object_data": object_data,
                }
            )
        # else:
        #     print(
        #         f"The text: {text} is not a candidate of game point with confidence: {confidence}"
        #     )

        transcription_list.append(object_data)
        transcription[frame_index] = object_data
    return discarded_detections


def separate_game_videos(video_input, games, desc, output_folder="games"):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the input video
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_input}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for game_index, game in enumerate(games):
        if not game:  # Skip empty games
            continue

        start_frame = game[0]["frame_index"]
        end_frame = game[-1]["frame_index"]
        if desc:
            start_frame, end_frame = end_frame, start_frame

        # Set the video to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Create a VideoWriter object
        output_path = os.path.join(output_folder, f"game_{game_index + 1}.mp4")
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        for frame_index in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        print(f"Game {game_index + 1} saved to {output_path}")

    cap.release()
    print("All game videos have been separated and saved.")


def debug_save_data_to_debug(frame_index, current_frame, detections, fps):
    filename = f"tmp/frame_index_{frame_index}_timestamp_{str(datetime.timedelta(seconds=int(frame_index / fps)))}.jpg"
    save_frame(draw_bboxes_in_image(current_frame, detections), filename)
    save_frame(draw_bboxes_in_image(current_frame, detections), "tmp/here.jpg")


# Function to detect and track scoreboard
def find_scoreboard(video_input, jump=2, in_seconds=True, desc=False):
    local_folder = "./tmp"
    # Check if we have saved candidates and games for this video
    candidates_points = load_data(
        video_input, data_type="candidates", local_folder=local_folder
    )
    games = load_data(video_input, data_type="games", local_folder=local_folder)
    if candidates_points is None or games is None:
        # List to store the paths of images with detected numbers
        saved_images = []
        transcription_list = []
        transcription = {}
        candidates_points = {}
        candidates_games = {}
        target_scores = ["40", "30", "15", "0"]
        discarded_detections = []

        # Iterate through the video frames
        for frame_index, current_frame in get_frames(
            video_input, jump=jump, in_seconds=in_seconds, desc=desc
        ):
            # Run your OCR detection logic
            detections = reader.readtext(current_frame)
            frame_discarded_detections = organize_detections(
                detections,
                frame_index=frame_index,
                jump=jump,
                candidates=candidates_points,
                threshold=0.75,
                transcription_list=transcription_list,
                transcription=transcription,
                target_scores=target_scores,
            )
            discarded_detections = discarded_detections + frame_discarded_detections
            print(f"frame_index: {frame_index}")
            print(
                f"timestamp: { str(datetime.timedelta(seconds=int(frame_index / fps)))}"
            )
            # debug_save_data_to_debug(frame_index, current_frame, detections, fps)
            # debug_save_data_to_debug(frame_index, current_frame, [t.values() for t in transcription_list], fps,)
            # if candidates_points:
            #     last_candidates = [
            #         (
            #             obj["object_data"]["bbox"],  # bbox
            #             obj["object_data"]["text"],  # text
            #             obj["object_data"]["confidence"],  # confidence
            #         )
            #         for obj in candidates_points[max(candidates_points.keys())]
            #     ]
            #     debug_save_data_to_debug(
            #         frame_index,
            #         current_frame,
            #         last_candidates,
            #         fps,
            #     )

            print("")

        # Save candidates and games for future use
        save_data(
            video_input,
            candidates_points,
            data_type="candidates",
            local_folder=local_folder,
        )

        # Separate in possible games
        games = get_games(candidates_points, desc)

        save_data(video_input, games, data_type="games", local_folder=local_folder)
        save_data(
            video_input,
            games,
            data_type="discarded_detections",
            local_folder=local_folder,
        )
        save_data(
            video_input, games, data_type="transcription", local_folder=local_folder
        )
        save_data(
            video_input,
            games,
            data_type="transcription_list",
            local_folder=local_folder,
        )
    else:
        print("Candidates and games already exist, skipping processing...")

    # Separate game videos
    separate_game_videos(video_input, games, desc)
    # Iterate here with candidates
    # find_points_trusted_coordinates(candidates_points, desc)
    # Analyze the scoreboard after processing all frames
    # games = analyze_scoreboard(candidates_points, desc)
    return "a"


def group_detections_by_score(detections, target_scores):
    # Group detections by score
    lists_by_score = {score: [] for score in target_scores}
    all_scores = []
    for detection in detections:
        text = detection["object_data"]["text"]
        if text in target_scores:
            lists_by_score[text].append(detection)
            all_scores.append(detection)
    return lists_by_score, all_scores


def get_games(candidates, desc):
    """This assumes that after 40 is game.
    And in fact is a gold point match

    Args:
        candidates (_type_): _description_
        desc (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("get_games")
    games = []
    sets = []
    target_scores = ["40", "30", "15", "0"]
    current_game = []
    on_change_create_new_game_score_1 = False
    on_change_create_new_game_score_2 = False
    is_a_frame_of_a_new_game_with_anomaly = False
    for frame_index, detections in sorted(candidates.items(), reverse=desc):
        lists_by_score, all_scores = group_detections_by_score(
            detections, target_scores
        )

        # Just analize simple cases
        is_simple_case = len(all_scores) == 2
        if not is_simple_case:
            print(f"WARNING: Complex case detected at frame {frame_index}.")
            continue
        print(
            f"frame_index: {frame_index} , timestamp {str(datetime.timedelta(seconds=int(frame_index / fps)))}, "
        )
        score_1 = int(all_scores[0]["object_data"]["text"])
        score_2 = int(all_scores[1]["object_data"]["text"])
        game_moment = {
            "score_1": score_1,
            "score_2": score_2,
            # "is_start_game": False, # we can detect this because detecting desc and if is the first or last game_moment
            # "is_end_game": False, # we can detect this because detecting desc and if is the first or last game_moment
            # desc: desc # we can detect this because if frame_index increase or decrease in a game
            "frame_index": frame_index,
        }
        # Detect end of a game if desc check with 0 if asc check with 40
        edge_value = 0 if desc else 40
        if score_1 == edge_value:
            on_change_create_new_game_score_1 = True
        if score_2 == edge_value:
            on_change_create_new_game_score_2 = True

        # New Game to track
        is_a_frame_of_a_new_game = (
            on_change_create_new_game_score_1
            and score_1 != edge_value
            or on_change_create_new_game_score_2
            and score_2 != edge_value
        )

        if len(current_game) > 0:
            prev_score_1 = current_game[-1]["score_1"]
            prev_score_2 = current_game[-1]["score_2"]
            # This means that "the game didn't finish", that the algorithm couldn't detect all texts.
            if desc:
                is_a_frame_of_a_new_game_with_anomaly = not (
                    prev_score_1 >= score_1 and prev_score_2 >= score_2
                )
            else:
                is_a_frame_of_a_new_game_with_anomaly = not (
                    prev_score_1 <= score_1 and prev_score_2 <= score_2
                )

        if is_a_frame_of_a_new_game:
            games.append(current_game)
            # Restart current_game_tracking
            current_game = [game_moment]
            on_change_create_new_game_score_1 = False
            on_change_create_new_game_score_2 = False
            is_a_frame_of_a_new_game_with_anomaly = False
        elif is_a_frame_of_a_new_game_with_anomaly:
            for g in current_game:
                g["has_anomaly"] = True
            games.append(current_game)
            # Restart current_game_tracking
            current_game = [game_moment]
            on_change_create_new_game_score_1 = False
            on_change_create_new_game_score_2 = False
            is_a_frame_of_a_new_game_with_anomaly = False
        else:
            current_game.append(game_moment)

    return games


def analyze_scoreboard(candidates, desc=False):
    games = []
    target_scores = ["40", "30", "15", "0"]
    current_game = []
    for frame_index, detections in sorted(candidates.items(), reverse=desc):

        lists_by_score = group_detections_by_score(detections, target_scores)

        analyze_frame_for_game_creation(
            lists_by_score["40"],
            lists_by_score["30"],
            lists_by_score["15"],
            lists_by_score["0"],
            candidates,
            desc,
            frame_index,
            current_game,
        )

        # if game has initial point and end point
        # games.append(game)

    return games


def analyze_frame_for_game_creation(
    list_of_40, list_of_30, list_of_15, list_of_0, candidates, desc, frame_index, game
):
    all_scores = list_of_40 + list_of_30 + list_of_15 + list_of_0
    is_simple_case = len(all_scores) == 2

    if is_simple_case:
        analyze_simple_case(
            list_of_40,
            list_of_30,
            list_of_15,
            list_of_0,
            candidates,
            desc,
            frame_index,
            game,
        )
    else:
        print(
            f"Complex case detected at frame {frame_index}. Manual review may be needed."
        )


def analyze_simple_case(
    list_of_40, list_of_30, list_of_15, list_of_0, candidates, desc, frame_index, game
):
    all_scores = list_of_40 + list_of_30 + list_of_15 + list_of_0

    score_1 = int(all_scores[0]["object_data"]["text"])
    score_2 = int(all_scores[1]["object_data"]["text"])
    game_moment = {
        "score_1": score_1,
        "score_2": score_2,
        "is_start_game": False,
        "is_end_game": False,
        "frame_index": frame_index,
    }

    # Simple case: a game started
    if len(list_of_15 + list_of_0) == 2:
        game.append(game_moment)
        if desc:
            return game  # This is the start of the game when analyzing in reverse

    # Simple case: a game will finish
    if desc:
        return game  # This is the end of the game when analyzing in reverse

    scores = [
        score
        for score_list in [list_of_40, list_of_30, list_of_15, list_of_0]
        for score in score_list
    ]

    if len(scores) == 2:
        game_moment = {
            "score_1": scores[0]["object_data"]["text"],
            "score_2": scores[1]["object_data"]["text"],
            "is_start_game": False,
            "is_end_game": "40"
            in [scores[0]["object_data"]["text"], scores[1]["object_data"]["text"]],
            "frame_index": frame_index,
        }
        game.append(game_moment)

    if desc:
        analyze_descretion(game)
    else:
        analyze_increase(game)
