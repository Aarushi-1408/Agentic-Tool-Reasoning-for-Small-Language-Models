import json
import random
from typing import List, Dict, Any, Optional

class TaskGenerator:
    """
    Synthetic training task generator.
    Generates novel task prompts across 5 behavior categories:
    - no_tool: General knowledge & reasoning without tools.
    - single_tool: Tasks requiring 1 tool call.
    - two_step: 2 sequential tool interactions.
    - multi_step: 3+ sequential tool interactions.
    - error_recovery: Tasks triggering real tool errors where model must recover.
    
    Training tasks are completely separate from frozen evaluation tasks.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def generate_tasks(
        self,
        num_no_tool: int = 250,
        num_single_tool: int = 75,
        num_two_step: int = 75,
        num_multi_step: int = 75,
        num_error_recovery: int = 25,
        id_prefix: str = "train_v4_",
    ) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []

        # 1. No-Tool Templates (Novel training tasks, zero overlap with eval_v1.json)
        no_tool_templates = [
            ("What does CPU stand for in computer hardware?", "easy", False),
            ("What is the freezing point of water in Celsius?", "easy", False),
            ("What currency is used in the United Kingdom?", "easy", False),
            ("Who painted the Starry Night?", "easy", False),
            ("What is the chemical symbol for Gold?", "easy", False),
            ("How many sides does a hexagon have?", "easy", False),
            ("What is the largest ocean on Earth?", "easy", False),
            ("Explain the basic difference between RAM and ROM.", "medium", False),
            ("What planet is known as the Red Planet?", "easy", False),
            ("What is the speed of light in vacuum approximately in km/s?", "medium", False),
            ("Who proposed the theory of general relativity?", "easy", False),
            ("What is the primary gas found in Earth's atmosphere?", "easy", False),
            ("What does HTTP stand for in networking?", "easy", False),
            ("In Python, what is the difference between a list and a tuple?", "medium", False),
            ("What is the capital of Germany?", "easy", False),
            ("How many bits are in a single byte?", "easy", False),
            ("What is the square root of 64?", "easy", False),
            ("Name the process by which plants convert sunlight into energy.", "easy", False),
            ("What is the smallest prime number?", "easy", False),
            ("What does GPU stand for?", "easy", False),
            ("Which continent is Egypt primarily located on?", "easy", False),
            ("What is the capital of Australia?", "easy", False),
            ("What is the chemical formula for carbon dioxide?", "easy", False),
            ("How many colors are in a standard rainbow?", "easy", False),
            ("What is the main function of red blood cells?", "medium", False),
            ("What does HTML stand for?", "easy", False),
            ("Who wrote the play Hamlet?", "easy", False),
            ("What is 12 squared?", "easy", False),
            ("What is the capital of Canada?", "easy", False),
            ("What is the largest organ in the human body?", "easy", False),
            ("What force keeps planets in orbit around the Sun?", "easy", False),
            ("How many degrees are in a right angle?", "easy", False),
            ("What language is spoken in Brazil?", "easy", False),
            ("What does URL stand for?", "easy", False),
            ("What is the hardest natural substance on Earth?", "easy", False),
            ("What planet is closest to the Sun?", "easy", False),
            ("What does API stand for in software engineering?", "easy", False),
            ("Explain what recursion means in computer science in simple terms.", "medium", False),
            ("What is the chemical formula for table salt?", "easy", False),
            ("How many continents are there on Earth?", "easy", False),
            ("Is an apple a fruit or a vehicle?", "easy", False),
            ("Define the term 'algorithm' in computer science.", "medium", False),
            ("What is the capital of France?", "easy", False),
            ("How many hours are in a standard day?", "easy", False),
            ("What is the boiling point of water at sea level in Celsius?", "easy", False),
            ("Which planet is known as the Ringed Planet?", "easy", False),
            ("What does TCP stand for in networking?", "easy", False),
            ("What is the author of '1984'?", "easy", False),
            ("What is the speed of sound in air approximately in m/s?", "medium", False),
            ("What is 15 minus 7?", "easy", False),
        ]

        # 2. Single-Tool Templates
        single_tool_templates = [
            ("Calculate 29 * 43.", "single_tool", "easy", True),
            ("Calculate (180 + 320) / 10.", "single_tool", "easy", True),
            ("Calculate 169 / 13.", "single_tool", "easy", True),
            ("Calculate 888 - 321.", "single_tool", "easy", True),
            ("Calculate 18 * 14.", "single_tool", "easy", True),
            ("Calculate 2048 / 64.", "single_tool", "easy", True),
            ("Calculate 315 + 485.", "single_tool", "easy", True),
            ("Echo the string 'Hello Synthetic Dataset'.", "single_tool", "easy", True),
            ("Echo the string 'Agentic Data Scaling'.", "single_tool", "easy", True),
            ("Echo the string 'Qwen3 Model Training'.", "single_tool", "easy", True),
            ("Echo the text 'System Verification OK'.", "single_tool", "easy", True),
            ("Echo the text 'Pipeline Generation Complete'.", "single_tool", "easy", True),
            ("Retrieve the atomic number of gold with key 'gold_atomic_number'.", "single_tool", "easy", True),
            ("Retrieve the value for 'tokyo_population' from database.", "single_tool", "easy", True),
            ("Query the database for value 'paris_berlin_distance_km'.", "single_tool", "easy", True),
            ("Look up key 'orange_price' in the store inventory database.", "single_tool", "easy", True),
            ("Query the database for key 'moon_distance_km'.", "single_tool", "easy", True),
            ("Fetch the exchange rate for key 'usd_to_eur'.", "single_tool", "easy", True),
            ("Reverse the string 'AntigravitySystem'.", "single_tool", "easy", True),
            ("Reverse the string 'DeepMind'.", "single_tool", "easy", True),
            ("Reverse the string 'Transformers'.", "single_tool", "easy", True),
            ("Find the length of the string 'Antidisestablishmentarianism'.", "single_tool", "easy", True),
            ("Find the length of the string 'Deep Learning System'.", "single_tool", "easy", True),
            ("Convert the string 'cynaptics' to uppercase.", "single_tool", "easy", True),
            ("Convert the string 'HELLOMODEL' to lowercase.", "single_tool", "easy", True),
            ("Calculate 85 * 9.", "single_tool", "easy", True),
            ("Calculate 1024 - 512.", "single_tool", "easy", True),
            ("Echo the text 'Tool Execution Test'.", "single_tool", "easy", True),
            ("Reverse the string 'SyntheticData'.", "single_tool", "easy", True),
            ("Find the length of the string 'OpenSourceAI'.", "single_tool", "easy", True),
        ]

        # 3. Two-Step Templates
        two_step_templates = [
            ("Fetch 'apple_price' and 'orange_price' from database, then calculate apple_price + orange_price + 5.", "two_step", "medium", True),
            ("Look up 'moon_distance_km' and 'speed_of_light_km_s', then calculate how many seconds light takes to travel from the moon to Earth.", "two_step", "medium", True),
            ("Convert the string 'machine learning' to uppercase and then echo the resulting string.", "two_step", "medium", True),
            ("Find the length of 'Antidisestablishmentarianism' using string_manipulator, then multiply that length by 10 using calculator.", "two_step", "medium", True),
            ("Look up 'gold_atomic_number' in database, then calculate gold_atomic_number * 7.", "two_step", "medium", True),
            ("Look up 'tokyo_population', then divide it by 2000 using calculator.", "two_step", "medium", True),
            ("Look up 'paris_berlin_distance_km' and add 150 km to it.", "two_step", "medium", True),
            ("Convert 'autonomous agent' to uppercase using string_manipulator, then reverse the result.", "two_step", "medium", True),
            ("Find length of string 'AgenticToolReasoning', then echo the length integer.", "two_step", "medium", True),
            ("Look up 'orange_price', then calculate orange_price * 25 to find cost of 25 oranges.", "two_step", "medium", True),
            ("Look up 'orange_price', then add 10 to it using calculator.", "two_step", "medium", True),
            ("Reverse string 'PythonCode', then convert the reversed string to uppercase.", "two_step", "medium", True),
            ("Calculate 15 * 8 using calculator, then echo the result.", "two_step", "medium", True),
        ]

        # 4. Multi-Step Templates (3-step and 4-step sequential chains)
        multi_step_templates = [
            ("Look up 'apple_price' and 'orange_price', calculate total cost for 5 apples and 4 oranges, then reverse the total cost digits.", "multi_step", "hard", True),
            ("Look up 'gold_atomic_number', multiply by 15 using calculator, then reverse the digits of that product string.", "multi_step", "hard", True),
            ("Look up 'paris_berlin_distance_km', add 300 using calculator, divide by 5, then reverse the resulting digits.", "multi_step", "hard", True),
            ("Look up 'apple_price', 'orange_price', and 'usd_to_eur'. Calculate (apple_price + orange_price) * usd_to_eur * 10, and reverse the resulting number string.", "multi_step", "hard", True),
            ("Look up 'tokyo_population', divide by 1000 using calculator, convert resulting number string to uppercase (if string), then echo it.", "multi_step", "hard", True),
            ("Look up 'gold_atomic_number', add 21 to it using calculator, reverse the sum string, and echo the reversed string.", "multi_step", "hard", True),
            ("Look up 'apple_price' and 'orange_price', calculate total cost for 3 apples and 2 oranges, then reverse the total cost string.", "multi_step", "hard", True),
            ("Find length of string 'AntigravityAgent', multiply that length by 5 using calculator, then reverse the result string.", "multi_step", "hard", True),
            ("Look up 'moon_distance_km', divide by 100 using calculator, reverse the resulting string, and echo the reversed result.", "multi_step", "hard", True),
            ("Convert 'SyntheticDataset' to uppercase, find its length, multiply the length by 3, then echo the final result.", "multi_step", "hard", True),
            ("Look up 'orange_price', multiply by 15 using calculator, reverse the product string, then echo it.", "multi_step", "hard", True),
            ("Look up 'paris_berlin_distance_km', add 200 using calculator, divide by 5, then reverse the resulting digits.", "multi_step", "hard", True),
            ("Reverse 'MultiStepReasoning', convert to uppercase, find its length, then multiply length by 2.", "multi_step", "hard", True),
        ]

        # 5. Error Recovery Templates
        error_recovery_templates = [
            ("Look up 'saturn_distance_km' in the reference database. If that key is missing, look up 'moon_distance_km' instead and report its value.", "error_recovery", "hard", True),
            ("Evaluate expression '30 * 40 + bad_var'. If evaluation fails with an error, evaluate '30 * 40' instead and return the result.", "error_recovery", "hard", True),
            ("Try calling tool 'db_search' with key 'apple_price'. If 'db_search' is an unknown tool, call 'lookup' with key 'apple_price' instead.", "error_recovery", "hard", True),
            ("Try calculating expression with non-string argument expression=500. Upon receiving invalid argument error, fix the argument to string expression='500 / 20' and report the result.", "error_recovery", "hard", True),
            ("Look up non-existent key 'jupiter_radius_km'. If key is missing/returns error, look up 'gold_atomic_number' instead and report its atomic number.", "error_recovery", "hard", True),
            ("Try evaluating invalid math expression '100 / 0'. Upon receiving divide by zero error, evaluate '100 / 5' instead.", "error_recovery", "hard", True),
            ("Call string_manipulator with invalid operation 'translate' on text 'hello'. Upon receiving unsupported operation error, call operation 'uppercase' instead.", "error_recovery", "hard", True),
            ("Look up key 'invalid_key_999'. If lookup fails with missing key error, look up key 'tokyo_population' and report it.", "error_recovery", "hard", True),
        ]

        task_counter = 1

        def add_batch(templates, target_count, category):
            nonlocal task_counter
            for i in range(target_count):
                item = templates[i % len(templates)]
                prompt = item[0]
                diff = item[1] if len(item) > 1 else "medium"
                req_tool = item[2] if len(item) > 2 else True

                task_id = f"{id_prefix}{task_counter:03d}"
                task_counter += 1

                tasks.append({
                    "task_id": task_id,
                    "category": category,
                    "prompt": prompt,
                    "metadata": {
                        "intended_difficulty": diff,
                        "requires_tool": req_tool
                    }
                })

        add_batch(no_tool_templates, num_no_tool, "no_tool")
        add_batch(single_tool_templates, num_single_tool, "single_tool")
        add_batch(two_step_templates, num_two_step, "two_step")
        add_batch(multi_step_templates, num_multi_step, "multi_step")
        add_batch(error_recovery_templates, num_error_recovery, "error_recovery")

        return tasks
