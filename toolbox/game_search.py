from openai import OpenAI
import os
import re

import pandas as pd

######################## Parameters ########################
PROJECT_PATH = "/root/autodl-tmp/SoccerAgent"
client = OpenAI(api_key="", base_url="https://api.deepseek.com")

MAX_CANDIDATES_FOR_LLM = 12
MAX_CANDIDATES_HARD_LIMIT = 200


def workflow(input_text, Instruction, follow_up_prompt=None, max_tokens_followup=1500):
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": input_text}
        ],
        stream=False
    )

    first_round_reply = completion.choices[0].message.content

    if follow_up_prompt:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": Instruction},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": first_round_reply},
                {"role": "user", "content": follow_up_prompt}
            ],
            max_tokens=max_tokens_followup,
            stream=False
        )
        second_round_reply = completion.choices[0].message.content
        return first_round_reply, second_round_reply

    return first_round_reply


### GAME SEARCH

def extract_match_info(input_text):
    INSTRUCTION = """
    You are a helpful assistant that extracts structured information from natural language text about football matches. I will give you a sentence about a football match, and you need to extract the following information: league, season, date, time, and two teams. The output must strictly follow the format below:
    league: (england_epl, germany_bundesliga, europe_uefa-champions-league, italy_serie-a, france_league-1, spain_laliga, or unknown)
    season: xxxx-xxxx
    date: xxxx-xx-xx
    year: xxxx
    month: xx
    day: xx
    time: xx:xx (which means when this game kick-off, not the game timestamp of certain event)
    score: x - x (if score is not determined, write 'unknown' for only in this attribute)
    team1: yyy
    team2: yyy

    All above 'x' means a digit!! 'yyy' means a string.

    To be noted, if you can determine only one team, please assign the team to team1 and leave team2 as 'unknown'. If any information is missing or uncertain, write 'unknown'. You have to use the exactly same name of teams as provided in the input text. Do not output any other words.
    For other attributes, if any information is missing or uncertain, write 'unknown'. As for date, you should record in the form of xxxx-xx-xx if you can get the clear date; Meanwhile, as for year, month, day, you need capture as more information point to this game as possible, including year, month, and day, and record them in numbers.
    Do not guess any information. For example if year is not said clearly, don't guess the year through season. Only use the information provided in the input text. Do not output any other words.
    """

    default_dict = {
        "league": "unknown",
        "season": "unknown",
        "date": "unknown",
        "year": "unknown",
        "month": "unknown",
        "day": "unknown",
        "time": "unknown",
        "score": "unknown",
        "team1": "unknown",
        "team2": "unknown",
    }

    llm_output = workflow(input_text, INSTRUCTION)
    pattern = re.compile(r"^(\w+):\s*(.*?)\s*$", re.MULTILINE)
    match = pattern.findall(llm_output)
    info = {key: value for key, value in match}

    if not info:
        return default_dict

    for key, value in default_dict.items():
        info.setdefault(key, value)

    return info


def _safe_int(value):
    try:
        return int(str(value).lstrip('0') or '0')
    except Exception:
        return None


def _normalize_text(value):
    return str(value or "").strip().lower()


def _team_known(value):
    v = _normalize_text(value)
    return v not in {"", "unknown", "none", "null"}


def _score_row(row, info):
    score = 0

    if _normalize_text(info.get("league")) == _normalize_text(row.get("league")):
        score += 3
    if _normalize_text(info.get("season")) == _normalize_text(row.get("season")):
        score += 3
    if _normalize_text(info.get("date")) == _normalize_text(row.get("date")):
        score += 5

    year = _safe_int(info.get("year"))
    month = _safe_int(info.get("month"))
    day = _safe_int(info.get("day"))
    if year is not None and year == _safe_int(row.get("year")):
        score += 1
    if month is not None and month == _safe_int(row.get("month")):
        score += 1
    if day is not None and day == _safe_int(row.get("day")):
        score += 1

    time = _normalize_text(info.get("time"))
    if time and time != "unknown" and time == _normalize_text(row.get("time")):
        score += 2

    row_home = _normalize_text(row.get("home_team")).replace(" ", "")
    row_away = _normalize_text(row.get("away_team")).replace(" ", "")

    team1 = _normalize_text(info.get("team1")).replace(" ", "")
    team2 = _normalize_text(info.get("team2")).replace(" ", "")

    if _team_known(info.get("team1")):
        if team1 in row_home or team1 in row_away:
            score += 3
    if _team_known(info.get("team2")):
        if team2 in row_home or team2 in row_away:
            score += 3

    return score


def retrieve_candidates(info, csv_path=os.path.join(PROJECT_PATH, "database/Game_dataset_csv/game_database.csv")):
    df = pd.read_csv(csv_path)

    conditions = []
    if _team_known(info.get("league")):
        conditions.append(df["league"].astype(str).str.lower() == _normalize_text(info["league"]))
    if _team_known(info.get("season")):
        conditions.append(df["season"].astype(str).str.lower() == _normalize_text(info["season"]))

    year = _safe_int(info.get("year"))
    month = _safe_int(info.get("month"))
    day = _safe_int(info.get("day"))
    if year is not None:
        conditions.append(df["year"] == year)
    if month is not None:
        conditions.append(df["month"] == month)
    if day is not None:
        conditions.append(df["day"] == day)

    time = _normalize_text(info.get("time"))
    if time and time != "unknown":
        conditions.append(df["time"].astype(str).str.lower() == time)

    initial_filtered_df = df[pd.concat(conditions, axis=1).all(axis=1)] if conditions else df

    team_conditions = []
    team1 = _normalize_text(info.get("team1")).replace(" ", "")
    team2 = _normalize_text(info.get("team2")).replace(" ", "")

    searchable = initial_filtered_df
    home_no_space = searchable["home_team"].astype(str).str.replace(" ", "", regex=False)
    away_no_space = searchable["away_team"].astype(str).str.replace(" ", "", regex=False)

    if _team_known(info.get("team1")) and _team_known(info.get("team2")):
        team_conditions.append(
            (home_no_space.str.contains(team1, case=False, na=False) & away_no_space.str.contains(team2, case=False, na=False)) |
            (home_no_space.str.contains(team2, case=False, na=False) & away_no_space.str.contains(team1, case=False, na=False))
        )
    elif _team_known(info.get("team1")):
        team_conditions.append(
            home_no_space.str.contains(team1, case=False, na=False) |
            away_no_space.str.contains(team1, case=False, na=False)
        )
    elif _team_known(info.get("team2")):
        team_conditions.append(
            home_no_space.str.contains(team2, case=False, na=False) |
            away_no_space.str.contains(team2, case=False, na=False)
        )

    candidates_with_team = searchable[pd.concat(team_conditions, axis=1).any(axis=1)] if team_conditions else searchable

    return initial_filtered_df, candidates_with_team


def _limit_candidates_for_llm(candidates, info, limit=MAX_CANDIDATES_FOR_LLM):
    if candidates is None or len(candidates) <= limit:
        return candidates

    scored = candidates.copy()
    scored["_rank_score"] = scored.apply(lambda row: _score_row(row, info), axis=1)
    scored = scored.sort_values(by=["_rank_score"], ascending=False).head(limit)
    return scored.drop(columns=["_rank_score"])


def _deterministic_best_path(candidates, info):
    if candidates is None or len(candidates) == 0:
        return None

    scored = candidates.copy()
    scored["_rank_score"] = scored.apply(lambda row: _score_row(row, info), axis=1)
    best = scored.sort_values(by=["_rank_score"], ascending=False).iloc[0]
    return best["file_path"]


def finalize_candidate_selection(candidates, candidates_with_team, info, question):
    if candidates is None or len(candidates) == 0:
        return "We did not find the match you mentioned in the database."

    preferred = candidates_with_team if candidates_with_team is not None and len(candidates_with_team) > 0 else candidates

    if len(preferred) == 1:
        file_path = preferred.iloc[0]["file_path"]
        return f"The game information file path is: {file_path}"

    if len(preferred) > MAX_CANDIDATES_HARD_LIMIT:
        best = _deterministic_best_path(preferred, info)
        return (
            "The candidate pool is too large for LLM disambiguation. "
            f"Using deterministic ranking, the most probable game information file path is: {best}"
        )

    llm_candidates = _limit_candidates_for_llm(preferred, info, limit=MAX_CANDIDATES_FOR_LLM)

    if llm_candidates is None or len(llm_candidates) == 0:
        best = _deterministic_best_path(preferred, info)
        return f"The given information is vague. The most probable game information file path is: {best}"

    prompt = f"""
    You are a helpful assistant that selects the most likely match from a list of candidates based on the given information.
    We need to retrieve a file path for the most probable match from the question: "{question}".

    Parsed query info:
    {info}

    Candidate matches (top-ranked and truncated):
    """

    for i, row in llm_candidates.reset_index(drop=True).iterrows():
        prompt += f"""
        Candidate {i + 1}:
        - League: {row['league']}
        - Season: {row['season']}
        - Date: {row['date']}
        - Year: {row['year']}
        - Month: {row['month']}
        - Day: {row['day']}
        - Time: {row['time']}
        - Score: {row['score']}
        - Home Team: {row['home_team']}
        - Away Team: {row['away_team']}
        - file_path: {row['file_path']}
        """

    prompt += """
    Return ONLY one of the following:
    1) The exact file path if one candidate is clearly best.
    2) If ambiguous, return the most probable file path and a short note saying ambiguity remains.
    Keep the answer concise.
    """

    return workflow(prompt, "You are a soccer expert that selects the most likely match from candidate matches.")


def GAME_SEARCH(query, materials=None):
    info = extract_match_info(query)
    candidates, candidates_with_team = retrieve_candidates(info)
    return finalize_candidate_selection(candidates, candidates_with_team, info, query)
