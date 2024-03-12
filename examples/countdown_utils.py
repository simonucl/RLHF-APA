'''
Utility functions for countdown.py
'''
import re

def combine_nums(a, b):
    # Implicitly makes assumptions about the order of operations and valid operations
    a = int(a)
    b = int(b)
    possible = [[a+b, f"{a}+{b}={a+b}"], [a*b, f"{a}*{b}={a*b}"]]
    if a <= b:
        possible.append([b-a, f"{b}-{a}={b-a}"])
        if a != 0 and b % a == 0:
            possible.append([b//a, f"{b}/{a}={round(b//a,0)}"])
    else:
        possible.append([a-b, f"{a}-{b}={a-b}"])
        if b != 0 and a % b == 0:
            possible.append([a//b, f"{a}/{b}={round(a//b,0)}"])
    return possible

class CountdownNode:
    def __init__(self, idx, parent, nums, operations, heuristic):
        self.nums = nums
        self.operations = operations
        self.heuristic = heuristic
        self.parent = parent
        self.idx = idx
    
    def __lt__(self, other):
        return self.heuristic < other.heuristic


# Heuristics functions
def sum_heuristic(nums, target):
    if len(nums) == 1:
        return abs(nums[0] - target)
    return sum(abs(num - target) for num in nums) / len(nums)

def mult_heuristic(nums, target):
    # get closer to factors of target
    # return sum([1 if (nums[i] == 0 or target % nums[i] == 0 or nums[i] % target == 0) else 0 for i in range(len(nums))])
    # softer version, with distance to factors
    factors = [i for i in range(2, target+1) if target % i == 0]
    return sum([min(abs(num - factor) for factor in factors) for num in nums])
    
# prune functions
def great_prune(heuristic, target):
    # Simple pruning based on result magnitude
    return heuristic > target

def mult_prune(result, target):
    # Prune if result is not close to a factor of target
    factors = [i for i in range(1, target+1) if target % i == 0]
    return all(abs(result - factor) > target for factor in factors)

def simple_rating(search_path):
    # Simple rating function based on number of operations
    nodes_explored = search_path.count("Exploring Operation")
    return nodes_explored


def get_target_nums(search_path, mode=""):
    # Extracting the target and initial numbers from the first line
    first_line = search_path.strip().split('\n')[0]
    search_path = search_path.replace("<|endoftext|>","")
    if mode == "dt":
        first_line = first_line.split("->")[1]
    target_nums_match = re.match(r"Current State: (\d+):\[(.*?)\]", first_line)
    if not target_nums_match:
        return "Invalid input: Cannot find the initial state in the first line."
    
    target, nums = int(target_nums_match.group(1)), [int(n) for n in target_nums_match.group(2).split(", ")]
    return target, nums

def parse_trajectory(search_path, mode="dt"):
    # Extracting the target and initial numbers from the first line
    first_line = search_path.strip().split('\n')[0]
    search_path = search_path.replace("<|endoftext|>","")

    if mode == "dt":
        first_line = first_line.split("->")[1]
    target_nums_match = re.match(r"Current State: (\d+):\[(.*?)\]", first_line)
    if not target_nums_match:
        return "Invalid input: Cannot find the initial state in the first line."
    
    target, nums = int(target_nums_match.group(1)), [int(n) for n in target_nums_match.group(2).split(", ")]

    # Extract the operations from the line that claims the goal is reached.
    goal_lines = re.finditer(r"\d+,\d+ equal: Goal Reached", search_path)
    goal_lines = list(goal_lines)
    if not goal_lines:
        return "No goal reached statement found."

    goal_line = goal_lines[0]
    # get the last operation line before the goal reached statement
    operations = re.findall(r"Exploring Operation: (.*?=\d+), Resulting Numbers: \[(.*?)\]", search_path[:goal_line.start()])
    if not operations:
        return "No operations found leading to the goal."

    final_operation = operations[-1][0]
    try:
        predicted_result = int(final_operation.split('=')[1])
    except:
        print("couldnt parse last op", final_operation)
        return "Couldnt parse last op"
    if predicted_result != target:
        return "Invalid path: Final operation does not result in target."

    # get the last current state, operations before the goal reached statement, and extract the operations
    operation_list = re.findall(r"Current State: \d+:\[.*?\], Operations: \[(.*?)\]", search_path[:goal_line.start()])[-1].split(', ')
    operation_list = [op.replace("'", "") for op in operation_list]
    operation_list += [final_operation]

    # Verify each operation and keep track of the numbers involved
    available_numbers = nums
    for operation in operation_list:
        # Verify the operation
        try:
            left, right = operation.split('=')
        except:
            return f"Could not split operation into lhs, rhs"
        try:
            if eval(left) != int(right):
                return f"Invalid operation: {operation}"
        except Exception as e:
            return f"Error in evaluating operation {operation}: {e}"
        # get the numbers involved
        used_numbers = re.findall(r"\d+", left)
        for n in used_numbers:
            if int(n) not in available_numbers:
                return f"Invalid operation: {operation}, number {n} not available in {available_numbers}"
            
        available_numbers = [n for n in available_numbers if n not in used_numbers]
        available_numbers.append(int(right))

    return "Valid path."

def metric_fn(search_path, mode="dt"):
    rating = parse_trajectory(search_path, mode=mode)
    if rating == "Valid path.":
        score = simple_rating(search_path)
        first_line = search_path.strip().split('\n')[0]
        if "->" in first_line:
            first_line = first_line.split("->")[1]
        target_nums_match = re.match(r"Current State: (\d+):\[(.*?)\]", first_line)
        target, nums = int(target_nums_match.group(1)), [int(n) for n in target_nums_match.group(2).split(", ")]
        if len(nums) == 2:
            # 2c2 x ops (4) = 4
            max_nodes = 4
        elif len(nums) == 3:
            # 3c2 x ops (4) x 2c2 x ops (4) = 48
            max_nodes = 48 
        elif len(nums) == 4:
            # 4c2 x ops (4) x 3c2 x ops (4) x 2c2 x ops (4) = 1152
            max_nodes = 1152
        elif len(nums) == 5:
            # 5c2 x ops (4) x 4c2 x ops (4) x 3c2 x ops (4) x 2c2 x ops (4) = 46080
            max_nodes = 46080
        return (max(1.-score/max_nodes, 0.0), rating)
    return (0., rating)

def reward_fn(samples, prompts, outputs):
    rewards = []
    for sample in samples:
        rating, _ = metric_fn(sample)
        rewards.append(rating)
    return rewards
    
if __name__ == "__main__":
    trajectory = """Current State: 47:[42, 61, 66], Operations: []
Exploring Operation: 66-42=24, Resulting Numbers: [61, 24]
Generated Node #0,0: 47:[61, 24] Operation: 66-42=24
Moving to Node #0,0
Current State: 47:[61, 24], Operations: ['66-42=24']
Exploring Operation: 61-24=37, Resulting Numbers: [37]
37,47 unequal: No Solution
Moving to Node #0,0
Current State: 47:[61, 24], Operations: ['66-42=24']
Exploring Operation: 61+24=85, Resulting Numbers: [85]
85,47 unequal: No Solution
Moving to Node #0
Current State: 47:[42, 61, 66], Operations: []
Exploring Operation: 61-42=19, Resulting Numbers: [66, 19]
Generated Node #0,1: 47:[66, 19] Operation: 61-42=19
Moving to Node #0,1
Current State: 47:[66, 19], Operations: ['61-42=19']
Exploring Operation: 66-19=47, Resulting Numbers: [47]
47,47 equal: Goal Reached47,47 equal: Goal Reached"""
    result = parse_trajectory(trajectory, mode="sft")
    print(result)
    
