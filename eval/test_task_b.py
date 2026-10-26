# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:58:07 2026

@author: Jack Street
"""

import os
import sys
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from logic_engine import LogicEngine

KB_PATH = os.path.join(PROJECT_ROOT, "kb", "task_b_kb.csv")
EVAL_PATH = os.path.join(PROJECT_ROOT, "eval", "task_b_eval_set.csv")


def normalise_logic_entity(entity_text: str) -> str:
    entity = entity_text.strip().lower()

    if entity.endswith(" is"):
        entity = entity[:-3].strip()

    if entity.endswith("_is"):
        entity = entity[:-3].strip()

    entity = entity.replace("-", "_")
    entity = entity.replace(" ", "_")
    return entity


def normalise_logic_property(prop_text: str) -> str:
    prop = prop_text.strip().lower().replace("-", " ").replace("_", " ")

    mapping = {
        "dairy": "Dairy",
        "gluten": "Gluten",
        "egg": "Egg",
        "eggs": "Egg",
        "nut": "Nut",
        "nuts": "Nut",
        "vegan": "Vegan",
        "sesame": "Sesame",
        "seed": "Seed",
        "dairy free": "DairyFree",
        "egg free": "EggFree",
        "gluten free": "GlutenFree",
        "nut free": "NutFree",
        "sesame free": "SesameFree",
        "not dairy free": "NotDairyFree",
        "not egg free": "NotEggFree",
        "not gluten free": "NotGlutenFree",
        "not nut free": "NotNutFree",
        "not sesame free": "NotSesameFree",
        "unsafe for milk allergy": "UnsafeForMilkAllergy",
        "unsafe for coeliac": "UnsafeForCoeliac",
        "unsafe for egg allergy": "UnsafeForEggAllergy",
        "unsafe for nut allergy": "UnsafeForNutAllergy",
        "unsafe for sesame allergy": "UnsafeForSesameAllergy",
    }

    return mapping.get(prop, prop_text.strip().replace(" ", ""))


def make_statement(entity_text: str, property_text: str) -> str:
    entity = normalise_logic_entity(entity_text)
    prop = normalise_logic_property(property_text)
    return f"{prop}({entity})"


def load_eval_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    logic_engine = LogicEngine()
    logic_engine.load_kb_from_csv(KB_PATH)

    ok, msg = logic_engine.kb_is_consistent()
    print("=== Task B KB Consistency Check ===")
    print(msg)
    print()

    if not ok:
        print("KB is inconsistent. Stop here and fix task_b_kb.csv first.")
        return

    rows = load_eval_rows(EVAL_PATH)

    total = 0
    correct = 0

    print("=== Task B Evaluation Results ===")

    for row in rows:
        mode = row["mode"].strip().upper()
        entity = row["entity"].strip()
        prop = row["property"].strip()
        expected = row["expected"].strip().upper()

        total += 1

        if mode == "CHECK":
            stmt = make_statement(entity, prop)
            result = logic_engine.check_statement(stmt)
            actual = result.status

            is_correct = (actual == expected)
            if is_correct:
                correct += 1

            print("-" * 70)
            print(f"Query: CHECK THAT {entity} IS {prop}")
            print(f"Statement: {stmt}")
            print(f"Expected: {expected}")
            print(f"Actual:   {actual}")
            print(f"Pass:     {is_correct}")
            print(f"Explanation: {result.explanation}")

        elif mode == "ADD":
            stmt = make_statement(entity, prop)
            added, msg = logic_engine.add_statement_if_consistent(stmt)
            actual = "ADDED" if added else "REJECTED"

            is_correct = (actual == expected)
            if is_correct:
                correct += 1

            print("-" * 70)
            print(f"Query: I KNOW THAT {entity} IS {prop}")
            print(f"Statement: {stmt}")
            print(f"Expected: {expected}")
            print(f"Actual:   {actual}")
            print(f"Pass:     {is_correct}")
            print(f"Explanation: {msg}")

        else:
            print("-" * 70)
            print(f"Skipping invalid row with mode={mode}")

    print()
    print("=== Summary ===")
    print(f"Passed: {correct}/{total}")
    print(f"Accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    main()