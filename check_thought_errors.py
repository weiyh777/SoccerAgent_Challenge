import os
import glob
import re
import json

def check_errors(thoughts_dir):
    if not os.path.exists(thoughts_dir):
        print(f"Directory not found: {thoughts_dir}")
        return

    files = glob.glob(os.path.join(thoughts_dir, "*.txt"))
    print(f"Found {len(files)} thought files in {thoughts_dir}")
    
    # Keywords that might indicate a crash or tool failure
    error_keywords = [
        # "Traceback (most recent call last)", 
        # "RuntimeError",
        # "ValueError",
        # "TypeError", 
        # "KeyError",
        # "IndexError",
        # "AttributeError",
        # "ImportError",
        # "CUDA out of memory",
        # "Tool execution failed",
        # "Max retries exceeded",
        # "Error occurred"
        "[STOP]"
    ]
    
    files_with_errors = []
    files_without_answer = []

    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for explicit python error tracebacks or error messages
            found_keywords = [kw for kw in error_keywords if kw in content]
            if found_keywords:
                files_with_errors.append((file_name, found_keywords))
            
            # Check if we can extract an option like O1, O2, etc.
            # Mirroring the logic in run_challenge_submission.py
            option_pattern = r'\b(O[1-9])\b'
            matches = re.findall(option_pattern, content.upper())
            
            if not matches:
                # If no Option is found, it's likely a failure to answer
                # Only add if not already in files_with_errors to avoid duplication, or keep separate?
                # Let's keep separate lists but maybe some overlap.
                files_without_answer.append(file_name)

        except Exception as e:
            print(f"Could not read {file_name}: {e}")

    # Report results
    print("-" * 50)
    if files_with_errors:
        print(f"Found {len(files_with_errors)} files with error keywords:")
        for name, kws in sorted(files_with_errors):
            print(f"  [{name}] contains: {', '.join(kws)}")
    else:
        print("No files containing standard error keywords found.")

    print("-" * 50)
    if files_without_answer:
        print(f"Found {len(files_without_answer)} files with no valid Option (O1-O9) output:")
        for name in sorted(files_without_answer):
            if not any(name == err_file[0] for err_file in files_with_errors):
                print(f"  [{name}] (No explicit error keywords)")
            else:
                print(f"  [{name}] (Also has error keywords)")
    else:
        print("All files contain a valid Option (O1-O9).")
    print("-" * 50)
    
    original_json_path = None
    output_json_path = None
    
    original_json_path = "/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge/challenge_with_tasks.json"
    output_json_path = "/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge/challenge_longchain.json"
    
    # Save hard questions if paths are provided
    if original_json_path and output_json_path:
        error_file_names = set()
        for fname, _ in files_with_errors:
            error_file_names.add(fname)
        for fname in files_without_answer:
            error_file_names.add(fname)
        
        error_ids = set()
        for fname in error_file_names:
            # Assuming filename is like "123.txt"
            base_name = os.path.splitext(fname)[0]
            # Handle cases where filename might not be just an ID? 
            # The prompt implies they correspond to IDs.
            # Usually IDs are integers in the challenge but filenames are strings.
            # Let's try to convert to int if possible to match JSON IDs which might be ints.
            if base_name.isdigit():
                error_ids.add(int(base_name))
            else:
                # If IDs are strings in json or filename is complex
                error_ids.add(base_name)
        
        print(f"Collecting hard questions for {len(error_ids)} failed IDs...")

        try:
            if not os.path.exists(original_json_path):
                 print(f"Original JSON not found: {original_json_path}")
            else:
                with open(original_json_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                hard_questions = []
                for item in original_data:
                    # Check both string and int representation just in case
                    if item['id'] in error_ids or str(item['id']) in error_ids:
                        hard_questions.append(item)
                
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(hard_questions, f, indent=4, ensure_ascii=False)
                    
                print(f"Saved {len(hard_questions)} hard questions to {output_json_path}")
            
        except Exception as e:
            print(f"Error saving hard questions: {e}")

if __name__ == "__main__":
    # Default path based on where run_challenge_submission.py puts it
    base_dir = "/root/autodl-tmp/SoccerAgent"
    thoughts_dir = os.path.join(base_dir, "agent_thoughts_0222_new")
    
    check_errors(thoughts_dir)
