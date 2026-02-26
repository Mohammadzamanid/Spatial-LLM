#!/usr/bin/env python3
"""Generate synthetic spatial reasoning data and save to JSON."""

import json
import os
import random

OBJECTS = [
    "table", "chair", "lamp", "door", "window", "book", "cup", "box",
    "shelf", "desk", "sofa", "plant", "clock", "mirror", "rug", "bed",
    "fridge", "stove", "sink", "counter", "closet", "pillow", "basket",
]

SPATIAL_RELATIONS = {
    "spatial_relation": [
        ("The {a} is to the left of the {b}.", "left"),
        ("The {a} is to the right of the {b}.", "right"),
        ("The {a} is above the {b}.", "above"),
        ("The {a} is below the {b}.", "below"),
        ("The {a} is in front of the {b}.", "front"),
        ("The {a} is behind the {b}.", "behind"),
        ("The {a} is next to the {b}.", "next to"),
        ("The {a} is on top of the {b}.", "on top"),
    ],
    "object_location": [
        ("The {a} is inside the {b}.", "inside"),
        ("The {a} is near the {b}.", "near"),
        ("The {a} is far from the {b}.", "far"),
        ("The {a} is between the {b} and the {c}.", "between"),
    ],
    "navigation": [
        ("Go north from the {a} to reach the {b}.", "north"),
        ("Go south from the {a} to reach the {b}.", "south"),
        ("Go east from the {a} to reach the {b}.", "east"),
        ("Go west from the {a} to reach the {b}.", "west"),
        ("Turn left at the {a} then go straight to the {b}.", "left then straight"),
        ("Turn right at the {a} then go straight to the {b}.", "right then straight"),
    ],
}

QUESTIONS = {
    "spatial_relation": [
        "Where is the {a} relative to the {b}?",
        "What is the spatial relationship between the {a} and the {b}?",
    ],
    "object_location": [
        "Where is the {a}?",
        "Can you locate the {a}?",
    ],
    "navigation": [
        "How do you get from the {a} to the {b}?",
        "What direction do you go from the {a} to the {b}?",
    ],
}


def generate_sample(task_type, difficulty=1):
    objs = random.sample(OBJECTS, min(3, len(OBJECTS)))
    a, b = objs[0], objs[1]
    c = objs[2] if len(objs) > 2 else objs[0]

    templates = SPATIAL_RELATIONS[task_type]
    template, answer = random.choice(templates)
    context = template.format(a=a, b=b, c=c)

    question_templates = QUESTIONS[task_type]
    question = random.choice(question_templates).format(a=a, b=b, c=c)

    # Add complexity for higher difficulty
    if difficulty >= 2:
        extra_objs = random.sample(OBJECTS, 2)
        extra_rel = random.choice(SPATIAL_RELATIONS["spatial_relation"])
        context += " " + extra_rel[0].format(a=extra_objs[0], b=extra_objs[1])
    if difficulty >= 3:
        extra_objs = random.sample(OBJECTS, 2)
        extra_rel = random.choice(SPATIAL_RELATIONS["navigation"])
        context += " " + extra_rel[0].format(a=extra_objs[0], b=extra_objs[1])

    return {
        "context": context,
        "question": question,
        "answer": answer,
        "task_type": task_type,
        "difficulty": difficulty,
    }


def generate_dataset(num_samples, split_name):
    task_types = list(SPATIAL_RELATIONS.keys())
    data = []
    for _ in range(num_samples):
        task_type = random.choice(task_types)
        difficulty = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        data.append(generate_sample(task_type, difficulty))
    return data


def main():
    random.seed(42)
    os.makedirs("data/processed", exist_ok=True)

    splits = {
        "spatial_train": 5000,
        "spatial_val": 1000,
        "spatial_test": 1000,
    }

    for name, count in splits.items():
        data = generate_dataset(count, name)
        path = f"data/processed/{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Generated {len(data)} samples -> {path}")

    print("\nDone! Data saved to data/processed/")


if __name__ == "__main__":
    main()
