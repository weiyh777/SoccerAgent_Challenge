import csv
import re, os
import ast
from openai import OpenAI
import json
from tqdm import tqdm
import argparse
import sys
sys.path.append('/root/autodl-tmp/SoccerAgent')



######################################
##                                  ##
##   PHASE 1 - Task Decomposition   ##
##                                  ##
######################################


######################## Parameters ########################
PROJECT_PATH = "/root/autodl-tmp/SoccerAgent"
from toolbox import *

toolbox_functions = {
    "Textual Entity Search": TEXTUAL_ENTITY_SEARCH,
    "Textual Retrieval Augment": TEXTUAL_RETRIEVAL_AUGMENT,
    "Game Search": GAME_SEARCH,
    "Game Info Retrieval": GAME_INFO_RETRIEVAL,
    "Match History Retrieval": MATCH_HISTORY_RETRIEVAL,
    "Entity Recognition": FACE_RECOGNITION,
    "Number Recognition": JERSEY_NUMBER_RECOGNITION,
    "Camera Detection": CAMERA_DETECTION,
    # "Segment": SEGMENT,
    "Shot Change": SHOT_CHANGE,
    # "Action Classifier": ACTION_CLASSIFICATION,
    # "Commentary Generation": COMMENTARY_GENERATION,
    "Jersey Color Relevant VQA": JERSEY_COLOR_VLM,
    "Vision Language Model": VLM,
    "Replay Grounding": REPLAY_GROUNDING,
    "Score and Time Recognition": SCORE_TIME_DETECTION,
    "Frame Selection": FRAME_SELECTION,
    "Foul Recognition": FOUL_RECOGNITION
}

import os
os.system('')


######################## Helper Functions ########################

def load_toolbox(file_path):
    descriptions = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        # reader = csv.DictReader(csvfile)
        reader = csv.DictReader(csvfile, quotechar='"', skipinitialspace=True)
        i = 0
        for row in reader:
            i += 1
            tool_name = row['name']
            ability = row['ability']
            query_input = row['query input']
            material_input = row['material input']
            output = row['output']
            remark = row['remark']
            
            description = f"=== Tool Description for TOOL{i} ===\n"
            # description += f"Name: TOOL{i}\n"
            description += f"Name: {tool_name}\n"
            description += f"Ability: {ability}\n"
            description += f"Query Input: {query_input}\n"
            description += f"material Input: {material_input}\n"
            description += f"Output: {output}\n"
            description += f"Remark: {remark}\n"
            
            descriptions.append(description)
    
    return descriptions

def load_toolbox_str(file_path=os.path.join(PROJECT_PATH, "toolbox.csv")):
    toolbox = ""
    for i in load_toolbox(file_path):
        toolbox += f"{i}\n"
    # print(toolbox)
    return toolbox


def csv_to_task_string(csv_path=os.path.join(PROJECT_PATH, "tasks.csv")):
    result = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file, quotechar='"', skipinitialspace=True)
        next(reader) 
        for i, row in enumerate(reader, start=1):
            if not row:  
                continue
                
            task_str = f"Task{i}: ​**{row[0]}** {row[1]}"
            
            chain = None
            for j in range(2, 5):
                if j < len(row) and row[j].strip():
                    chain = row[j]
                    break
            
            if chain:
                task_str += f"\nRecommended chain: {chain}"
            
            result.append(task_str + "\n")
    
    return "\n".join(result)

def generate_prompt(taskdecompositionprompt, query, additional_material):
    prompt = taskdecompositionprompt
    prompt += f"\nQuery: {query}\n"
    prompt += f"Addittional Material: {additional_material}\n"
    return prompt

def workflow(input_text, Instruction, follow_up_prompt=None, api_key="", max_tokens_followup=1500):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
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
    else:
        return first_round_reply
    
def parse_input(input_str):
    known_info = re.findall(r'\$(.*?)\$', input_str)
    
    tool_chain = re.findall(r'\*(.*?)\*', input_str)
    
    return (known_info, tool_chain)

def generate_prompt_execution(query, material, response, toolbox, options):
    prompt_execution = f"""As a multi-agent core in the Soccer Question Answering Assistant, you are required to execute the following tool chain to answer the question:

"{query}"

with the following additional material:

{material}

There are several options:

{options}

with the known info as:

{parse_input(response)[0]}

and you should execute the following tool chain to solve the question:

{parse_input(response)[1]}

As for the usage of the tools, you should follow the following references:

{toolbox}

For every tool above, we would input queries and materials into the tool for execution, the queries are in **text** form and the materials are in list with **file paths**. If no file path is suitable, you just write in 'None' You should determine the contents of materials and queries based on the context of the question, known info and tool descriptions.

For every steps of excution, you should return me with a clear statement of the goal of this step in the context of the overall analysis, the specific tool you are using, and the input variables you are using.

<Call>
    <Purpose>Brief, clear statement of this step’s goal in context of overall analysis</Purpose> 
    <Query>[Query/question here(string). IMPORTANT!!: Such query is highly relevant to the toolbox descriptions. you need to think carefully about your purpose this step and generate appropriate query.]</Query> 
    <Material>[Material list here(a string showing list form). Here as well, you need to think carefully considering the purpose and toolbox.]</Material>
    <Tool>[Tool name here(string)]</Tool>
</Call>

If it is the last step of the execution, you should return me with the following format:

<EndCall>
    <Purpose>Brief, clear statement of this step’s goal in context of overall analysis</Purpose> 
    <Query>[Query/question here(string)]</Query> 
    <Material>[Material list with file paths here(a string showing list form)]</Material>
    <Tool>[Tool name here(string)]</Tool>
</EndCall>

Every time you return me with the instruction as above, I will execute it and return you with the feedback of the execution in this format:

<StepResult>
    <Answer>[The results of this time's execution here(string)]</Answer>
</StepResult>

For every time of generation, you should follow the following rules:
    1. You should be clear about the tool name (must be chosen from toolbox), file path and query/question in the instruction. This part is important for me to understand the context of the execution. You cannot change any of the information in the instruction.
    2. If I have given you the feedback of the execution, you should analyze what you should write in the next call based on the feedback considering the tool chain I gave you and the task descriptions and tool descriptions. You should not repeat the same instruction again.
    3. If my prompt leaves you to generate the first call, you should directly return me with the call in the form from <> to </>. You should not add any other information in the instruction.
    4. Otherwise, if in the prompt I have given you some <StepResult>, you should consider the total process of the execution and continue to return me exactly with the form from <> to </>. You should not add any other information in the instruction.

Tool Priority Instruction: You must ONLY call the Textual Retrieval Augment tool if you have successfully obtained a valid JSON file path from the Textual Entity Search tool. If the search tool returns no results, "None", or "Not found", you are STRICTLY FORBIDDEN from calling Textual Retrieval Augment. Instead, you should rely on your internal knowledge or choose a different strategy.
Tool Priority Instruction: When analyzing videos or actions, always PRIORITIZE using the 'Action Classifier' and 'Commentary Generation' tools over the 'Vision Language Model'. These two tools utilize specialized soccer-specific models that provide more accurate and domain-expert knowledge compared to the general-purpose Vision Language Model. Only use the 'Vision Language Model' if the specialized tools cannot handle the specific request or have failed.

Once again, I repreat that the question is:

"{query}"

with the following additional material:

{material}

with the known info as:

{parse_input(response)[0]}

and you should execute the following tool chain to solve the question:

{parse_input(response)[1]}

The following is all our execution history, now you can start with your call of first step:
    """

    return prompt_execution

######################################
##                                  ##
##   PHASE 2 - Tool Execution       ##
##                                  ##
######################################


def parse_call_response(model_reply):
    tool = re.search(r'<Tool>(.*?)</Tool>', model_reply, re.DOTALL)
    query = re.search(r'<Query>(.*?)</Query>', model_reply, re.DOTALL)
    material = re.search(r'<Material>(.*?)</Material>', model_reply, re.DOTALL)

    if not tool or not query or not material:
        raise ValueError("Invalid <Call> format in model_reply")

    return tool.group(1).strip(), query.group(1).strip(), material.group(1).strip()

def execute_tool_call(tool_name, query, material, toolbox_functions):
    if tool_name not in toolbox_functions:
        raise ValueError(f"Tool '{tool_name}' not found in toolbox_functions")

    tool_function = toolbox_functions[tool_name]
    execution_retults = None
    try:
        execution_retults = tool_function(query, material)
    except Exception as e:
        execution_retults = f"Error occurred: {str(e)}"
    return f"<StepResult>\n    <Answer>{execution_retults}</Answer>\n</StepResult>"


def _should_replan(step_result):
    text = str(step_result or "").lower()
    failure_markers = [
        "error occurred",
        "runtime error",
        "traceback",
        "module not found",
        "no matching file found",
        "failed to read json file",
        "file not found",
        "not found",
        "invalid",
        "cuda out of memory",
        "out of memory",
        "insufficient memory",
    ]
    return any(marker in text for marker in failure_markers)


def _tool_blocked_by_policy(tool_name, used_tools):
    specialized_used = (
        "Action Classifier" in used_tools
        or "Commentary Generation" in used_tools
    )
    return tool_name == "Vision Language Model" and specialized_used


def _policy_block_step_result():
    return (
        "<StepResult>\n"
        "    <Answer>Error occurred: Policy block - 'Vision Language Model' is disabled in this chain after "
        "'Action Classifier' or 'Commentary Generation' has been used. Please replan with specialized tools.</Answer>\n"
        "</StepResult>"
    )


def _build_replan_prompt(question, previous_call, last_step_result):
    return f"""System Feedback: The previous step failed or returned unusable output.

Original question:
{question}

Your previous call:
{previous_call}

Execution feedback:
{last_step_result}

You MUST replan now and produce a different next tool call that avoids repeating the failed strategy.
Rules:
1. Do not repeat the same <Tool>/<Query>/<Material> combination.
2. If material paths may be invalid, choose a path-robust alternative tool or use another available material.
3. If textual retrieval path is missing, do not call Textual Retrieval Augment directly.
4. Keep strictly the XML schema with <Call> ... </Call> or <EndCall> ... </EndCall>.
"""

def generate_LLM_prompt(query):
    prompt = f"""According to the total process of above conversation, now give me your answer to the question '{query}'. You should retrun me with the following form:
    
Answer: [Your answer here]
Reasoning and Explanation: [Your reasoning and explanation here]

Please make sure that your answer is consistent with the total process of the execution and the tool chain I gave you. You should not change the tool chain and the process of the execution. You should only give me the answer and the reasoning and explanation of the answer without any other words.
    """
    return prompt



def _finalize_with_llm(client, conversation_history, query, total_process, reason_tag):
    llm_prompt = generate_LLM_prompt(query)
    conversation_history.append({"role": "user", "content": llm_prompt})
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=conversation_history
    )
    final_reply = completion.choices[0].message.content
    total_process += f"\n[{reason_tag}]\n"
    total_process += final_reply
    return total_process


def execute_tool_chain(input_text, toolbox_functions, Instruction="You are a helpful multi-agent assistant that can answer questions about soccer.", api_key="", max_steps=18, max_replans=4):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    # Initialize the conversation history with the system instruction and user input
    conversation_history = [
        {"role": "system", "content": Instruction},
        {"role": "user", "content": input_text}
    ]
    total_process = ""
    step_count = 0
    replan_count = 0
    original_question = input_text
    used_tools = set()
    last_call_signature = None
    repeated_call_count = 0
    latest_query = "the user's question"
    while step_count < max_steps:
        step_count += 1
        # Generate a response from the model
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=conversation_history
        )
        # Get the model's reply
        model_reply = completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": model_reply})
        total_process += model_reply

        try:
            # Check if the reply contains <EndCall>
            if "<EndCall>" in model_reply:
                # Handle EndCall format logic
                if "<Tool>LLM</Tool>" in model_reply or "<Tool>None</Tool>" in model_reply: 
                    # Special case where model signals end but might use dummy tool or LLM tool name
                    # Or standard flow where we want LLM final answer
                    pass 
                else: 
                     # Check if it is a valid function call structure even in EndCall
                     try:
                        tool, query, material = parse_call_response(model_reply)
                        if tool != "LLM":
                             material = ast.literal_eval(material) if material is not None else []
                             if _tool_blocked_by_policy(tool, used_tools):
                                  user_execution = _policy_block_step_result()
                             else:
                                  user_execution = execute_tool_call(tool, query, material, toolbox_functions)
                                  used_tools.add(tool)
                             total_process += user_execution
                             # After final tool execution, we might still want LLM summary, or just return
                             # The original logic seemed to want to return LLM summary after EndCall
                     except ValueError:
                         # If EndCall but invalid format, we might assume it's just trying to end
                         pass

                # Generate a prompt for the LLM to answer the question
                # Attempt to parse query from EndCall if possible, otherwise use original input?
                # The original code used 'query' variable which was from parse_call_response
                # We need to ensure 'query' is available. 
                
                try:
                    tool, query, material = parse_call_response(model_reply)
                except ValueError:
                    # If parsing fails in EndCall, we use a generic prompt or recent query
                    query = "the user's question" 

                latest_query = query
                return _finalize_with_llm(
                    client,
                    conversation_history,
                    latest_query,
                    total_process,
                    reason_tag="FINALIZED_BY_ENDCALL"
                )

            # Standard Tool Call
            tool, query, material = parse_call_response(model_reply)
            latest_query = query
            call_signature = f"{tool}|{query}|{material}"
            if call_signature == last_call_signature:
                repeated_call_count += 1
            else:
                repeated_call_count = 1
                last_call_signature = call_signature

            if repeated_call_count >= 3:
                return _finalize_with_llm(
                    client,
                    conversation_history,
                    latest_query,
                    total_process,
                    reason_tag="FORCED_FINALIZE_REPEATED_CALL"
                )

            if tool == "LLM":
                return _finalize_with_llm(
                    client,
                    conversation_history,
                    latest_query,
                    total_process,
                    reason_tag="FINALIZED_BY_LLM_TOOL"
                )

            material = ast.literal_eval(material) if material is not None else []
            if _tool_blocked_by_policy(tool, used_tools):
                user_execution = _policy_block_step_result()
            else:
                user_execution = execute_tool_call(tool, query, material, toolbox_functions)
                used_tools.add(tool)

            if _should_replan(user_execution) and replan_count < max_replans:
                replan_count += 1
                replan_prompt = _build_replan_prompt(original_question, model_reply, user_execution)
                conversation_history.append({"role": "user", "content": replan_prompt})
                total_process += user_execution
                total_process += f"[REPLAN_TRIGGERED {replan_count}/{max_replans}]"
                continue

            conversation_history.append({"role": "user", "content": user_execution})
            total_process += user_execution

        except ValueError as e:
            # Format error - Ask model to retry
            # print(f"Format error detected, asking for retry: {e}")
            retry_prompt = f"System Error: Your response did not follow the required XML format (<Tool>, <Query>, <Material>). Please retry and strictly follow the format."
            conversation_history.append({"role": "user", "content": retry_prompt})
            continue # Retry the loop
        except Exception as e:
            print(f"Runtime error in tool execution loop: {e}")
            if replan_count < max_replans:
                replan_count += 1
                runtime_replan_prompt = _build_replan_prompt(original_question, model_reply, f"Runtime error in tool execution loop: {e}")
                conversation_history.append({"role": "user", "content": runtime_replan_prompt})
                total_process += f"\n[RUNTIME_REPLAN_TRIGGERED {replan_count}/{max_replans}] {e}\n"
                continue
            break # Break directly on other errors to avoid infinite loops

    total_process += f"\n[STOP] Reached step/replan budget. step_count={step_count}, replan_count={replan_count}\n"
    return total_process

######################## Some Basic Prompt Information ########################

toolbox_descriptions = load_toolbox_str()
tasks = csv_to_task_string()

TaskDecompositionPrompt = f"""
# Soccer Question Answering Assistant

## Task overview
You are a multi-modal agent that can answer questions about soccer knowledge.
For each question, you will reveive:
- A question about soccer considering different aspects of soccer
- You might also receive one or more video clip or image as context

Your task involves three sequential parts:
1. Problem Decomposition (Part 1) 
    - Identify available information 
    - Break down the question into sequential steps 
2. Sequential Tool Application (Part 2) 
    - Execute one tool at a time 
    - Record each tool’s output 
    - Continue until sufficient information is gathered 
3. Solution Synthesis (Part 3) 
    - Integrate all results 
    - Generate final answer


## Available Tools

For all the QA, you need to decompose them and 
Here are the tools that you can use to answer the questions:
{toolbox_descriptions}

## Common QA Tasks

Here are some common QA tasks that you might meet in the questions, for each types of questions, we provide the recommended tool chain for you to answer the questions:

{tasks}

To be noted, at this stage you only need to treat this question as open-ended QA task, you can use the common QA tasks as reference to decompose the question and identify the required tools.

## Response Format for Part 1
For each query, you should respond ONLY with: 
    Known Info: [list any categories explicitly mentioned in the query and material] 
    Tool Chain: [list required tools connected by ->]

## Examples

Query 1: "How does the viewpoint of the camera shift in the video?"
Addittional Material: "video": ["/remote-home/jiayuanrao/MatchRAG/test_QA/lzf/Q8/QA_material/video_once/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_135.mkv"]

Your response:
Known Info: [$VideoClip$]
Tool Chain: [*Shot Change* -> *Crop Video* -> *Camera Detection* -> *LLM*]

Query 2: "What was the final score of the game 2015-02-21 - 18-00 Chelsea vs Burnley?"
Addittional Material: None

Your response:
Known Info: [$GameContext$]
Tool Chain: [*Game Search* -> *Game info Retrieval* -> *Match History Retrieval* -> *LLM*]

Query 3: "How many goals did the player who forced a corner score for Borussia Dortmund's senior team?"
Addittional Material: "video": ["/remote-home/jiayuanrao/dataset/matchtime/video_clips/SN-Caption-test-align/germany_bundesliga_2015-2016/2015-11-08 - 17-30 Dortmund 3 - 2 Schalke/1_42_03.mp4"]

Your response:
Known Info: [$VideoClip$, $GameContext$]
Tool Chain: [*Vision Language Model* -> *Entity Recognition* -> *Text Retrieval Augment* -> *LLM*]

## Important Rules
1. You should only use the tools provided in the toolbox to answer the questions and provide the exact tool names.
2. Use exact item category names with $$ to represent the information categories.
3. Use exact tool category names with ** as shown above to represent the tools.
4. Only respond with Part 1 analysis - Parts 2 & 3 will be addressed in subsequent interactions.
5. Connect tools using -> symbol
6. Try your best to decompose the question and identify the required tools, you can first reference the common QA tasks to get some ideas. If the template fits the question, you can directly use the recommended tool chain. If not, you can try to decompose the question and identify the required tools.
"""
def EXECUTE_TOOL_CHAIN(query, material, options):
    prompt = generate_prompt(TaskDecompositionPrompt, query, material)
    res = workflow(input_text=prompt, Instruction="You are an expert in soccer.")
    result = execute_tool_chain(generate_prompt_execution(query, material, res, toolbox_descriptions, options), toolbox_functions)
    return result
