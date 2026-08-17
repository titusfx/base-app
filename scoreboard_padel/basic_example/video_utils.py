import os
import cv2


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
