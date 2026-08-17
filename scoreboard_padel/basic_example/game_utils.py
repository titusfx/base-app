import datetime


def group_detections_by_score(detections, target_scores):
    # Group detections by score
    lists_by_score = {score: [] for score in target_scores}
    all_scores = []
    for detection in detections:
        text = detection["object_data"]["text"].strip()
        if text in target_scores:
            lists_by_score[text].append(detection)
            all_scores.append(detection)
    return lists_by_score, all_scores


def is_number(text: str):
    try:
        int(text)
        return True
    except ValueError:
        return False


def group_detections_by_numbers(detections):
    # Group detections by score
    lists_by_number = {}
    for detection in detections:
        text = detection["object_data"]["text"].strip()
        if is_number(text):
            lists_by_number.setdefault(int(text), []).append(detection)

    return lists_by_number


def get_games(candidates, desc, fps):
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
    # Iterate over the candidates in reverse  or not, depending on desc (if analyze the video desc or asc)
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


class GameDetection:

    def __init__(self, desc=False, target_scores=["40", "30", "15", "0"]) -> None:
        self.games = []
        self.sets = []
        self.desc = desc
        self.target_scores = target_scores
        self.all_candidates = {}

        self.team_1 = {"score_game": 0, "sets": []}
        self.team_2 = {"score_game": 0, "sets": []}
        # let assume that if is row is one point on top of the other
        # Team 1: 30
        # Team 2: 15

        self.scoreboard_order = "row"  # "column"
        self.board_score_detector = PadelScoreboard(self.scoreboard_order, 1920, 1080)

    @property
    def is_new_match(self):
        return self.all_candidates == {}

    def get_teams_score(self, all_scores):
        # Detect if bbox 0 belongs to team0
        bbox0 = all_scores[0]["object_data"]["bbox"]
        bbox1 = all_scores[1]["object_data"]["bbox"]
        if self.board_score_detector.team1_area is None:
            self.board_score_detector.define_areas(
                all_scores[0]["object_data"]["bbox"],
                all_scores[1]["object_data"]["bbox"],
            )

        team1 = self.board_score_detector.determine_team(bbox0)
        team2 = self.board_score_detector.determine_team(bbox1)

        score_1 = int(all_scores[0]["object_data"]["text"])
        score_2 = int(all_scores[1]["object_data"]["text"])
        if team1 != 0:
            score_1 = int(all_scores[1]["object_data"]["text"])
            score_2 = int(all_scores[0]["object_data"]["text"])

        assert team1 == 0
        assert team2 == 1
        return score_1, score_2

    def analyze_candidates(
        self, candidates, desc=None, target_scores=None, sets=None, fps=25
    ):
        """
        Receive in an incremental way the candidates of the match
        """
        if desc is None:
            desc = self.desc
        if target_scores is None:
            target_scores = self.target_scores
        if sets is None:
            sets = self.sets

        current_game = []
        on_change_create_new_game_score_1 = False
        on_change_create_new_game_score_2 = False
        is_a_frame_of_a_new_game_with_anomaly = False

        candidates_in_order = sorted(candidates.items(), reverse=desc)

        for frame_index, detections in candidates_in_order:
            lists_by_numbers = group_detections_by_numbers(detections)
            sure_points = (
                lists_by_numbers[15] + lists_by_numbers[30] + lists_by_numbers[40]
            )
            # it should exist either two 15,30, 40, 0
            frame_has_anomaly = len(sure_points) > 2 or (
                len(sure_points) == 1 and len(lists_by_numbers[0]) == 0
            )

            if frame_has_anomaly:
                print(
                    f"It has anomaly at frame {frame_index}. Can be because is not showing scoreboard or because OCR didn't find"
                )
                continue

            # lists_by_score, all_scores = group_detections_by_score(
            #     detections, target_scores
            # )

            # def clean_scores_candidates(all_scores, lists_by_score):
            #     simple_case = len(all_scores) == 2
            #     if simple_case:
            #         return all_scores, lists_by_score

            #     if len(all_scores) > 2:
            #         lists_by_score["40"] == 2 or lists_by_score[
            #             "30"
            #         ] == 2 or lists_by_score["15"] == 2

            # Clean all_scores: If I have two of (40, 30 or 15) and 0. We discard 0 as a score
            # self.clean_scores_candidates(all_scores, lists_by_score)
            # Just analize simple cases
            # is_simple_case = len(all_scores) == 2 or (
            #     len(all_scores) == 3 and lists_by_score["0"] == 3
            # )
            # if not is_simple_case:
            #     print(f"WARNING: Complex case detected at frame {frame_index}.")
            #     continue
            print(
                f"frame_index: {frame_index} , timestamp {str(datetime.timedelta(seconds=int(frame_index / fps)))}, "
            )
            simple_case = (len(sure_points) + lists_by_numbers[0]) >= 2

            game_points = []
            for game_point in sorted(target_scores, reverse=True):
                game_points = game_points + lists_by_numbers[game_point]

            score_1, score_2 = self.get_teams_score(game_points)

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
                self.games.append(current_game)
                # Restart current_game_tracking
                current_game = [game_moment]
                on_change_create_new_game_score_1 = False
                on_change_create_new_game_score_2 = False
                is_a_frame_of_a_new_game_with_anomaly = False
            elif is_a_frame_of_a_new_game_with_anomaly:
                for g in current_game:
                    g["has_anomaly"] = True
                self.games.append(current_game)
                # Restart current_game_tracking
                current_game = [game_moment]
                on_change_create_new_game_score_1 = False
                on_change_create_new_game_score_2 = False
                is_a_frame_of_a_new_game_with_anomaly = False
            else:
                current_game.append(game_moment)

        return self.games


class PadelScoreboard:
    def __init__(self, scoreboard_order, image_width, image_height):
        self.scoreboard_order = scoreboard_order
        self.image_width = image_width
        self.image_height = image_height
        self.team1_area = None
        self.team2_area = None

    def define_areas(self, bb1_team1, bb2_team2):
        if self.scoreboard_order == "row":
            # Extend horizontally from x = 0 to image_width, keep y fixed
            self.team1_area = [
                [0, bb1_team1[0][1]],
                [self.image_width, bb1_team1[2][1]],
            ]
            self.team2_area = [
                [0, bb2_team2[0][1]],
                [self.image_width, bb2_team2[2][1]],
            ]
        elif self.scoreboard_order == "vertical":
            # Extend vertically from y = 0 to image_height, keep x fixed
            self.team1_area = [
                [bb1_team1[0][0], 0],
                [bb1_team1[2][0], self.image_height],
            ]
            self.team2_area = [
                [bb2_team2[0][0], 0],
                [bb2_team2[2][0], self.image_height],
            ]

    def determine_team(self, bbox):
        """Determine which team the given bbox belongs to based on overlap."""
        if self.scoreboard_order == "row":
            y_center = (
                bbox[0][1] + bbox[2][1]
            ) / 2  # Calculate the y-center of the bbox
            if self.team1_area[0][1] <= y_center <= self.team1_area[1][1]:
                return 0
            elif self.team2_area[0][1] <= y_center <= self.team2_area[1][1]:
                return 1
        elif self.scoreboard_order == "vertical":
            x_center = (
                bbox[0][0] + bbox[2][0]
            ) / 2  # Calculate the x-center of the bbox
            if self.team1_area[0][0] <= x_center <= self.team1_area[1][0]:
                return 0
            elif self.team2_area[0][0] <= x_center <= self.team2_area[1][0]:
                return 1
        return None  # Not matching any team


# How to detect games:
# 1-we can detect only 40, 30 and 15.
#   1.1-if two of them that is the score then that is the score.
#   1.1 - if exist one and then exist a zero that is the current score game.
#   else
#       1.2If more than two that's an error. and probably the game score is there ( but it should be possible all the previous combinations, we will need to save it and keep track until we can discard)
# 2-we need to detect other numbers,
# Is set tiebreak when sets are 6 and 6. in that way the analysis is different. Because we analyze until


# Lets assume we start the game with 0,0 and sets = []
# The first game
# we start analyzing:
# - we should find (0,0) and 1 of those 0 should convert to 15
# at some point it should happen one of the two things:
# Either appear a 1 after the game is won by 40,0
# or the other 0 should convert to 15, that means that in that game that area is the correct one
# This is assuming that the area of the game points may change shifting to the right or left
# In the 2nd game that shift doesn't happen it may happen when a set is finish
# A set is finished either difference of two games or golden game.
