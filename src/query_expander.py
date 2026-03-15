RESEARCH_SYNONYMS = {
    "training": ["reward", "optimization", "policy learning", "PPO"],
    "encouraged": ["reward shaping", "penalty", "incentive", "bonus"],
    "obstacle avoidance": ["collision avoidance", "safety reward", "dynamic obstacle"],
    "path planning": ["trajectory generation", "motion planning", "global path"],
    "dataset": ["training data", "image dataset", "data collection"],
    "detection": ["object detection", "yolo detection", "YOLOv10"],
    "smooth": ["trajectory smoothing", "b spline", "joint angle"],
    "goal": ["target point", "success condition", "reach"],
    "learn": ["reinforcement learning", "policy", "reward", "PPO training"],
    "collision": ["penalty", "obstacle", "avoidance", "safety"],
    "experiment": ["simulation", "setup", "MuJoCo", "evaluation"],
    "finding": ["conclusion", "result", "success rate", "performance"],
    "contribution": ["proposed", "novel", "framework", "key contribution"],
    "scenario": ["obstacle", "experiment", "path planning", "evaluation"],
    "noise": ["annealing", "gaussian", "exploration", "action noise"],
    "hardware": ["GPU", "configuration", "platform", "computation"],
    "images": ["dataset", "collected", "training", "split"],
    "limitation": ["conclusion", "degradation", "challenge", "future work"],
}

def expand_query(query):

    expanded = query

    for key, values in RESEARCH_SYNONYMS.items():

        if key in query.lower():

            expanded += " " + " ".join(values)

    return expanded