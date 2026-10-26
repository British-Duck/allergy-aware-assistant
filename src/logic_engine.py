# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:56:23 2026

@author: Jack Street
"""
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

from nltk.sem import Expression
from nltk.inference import ResolutionProver

read_expr = Expression.fromstring


@dataclass
class LogicKBResult:
    status: str          # CORRECT / INCORRECT / UNKNOWN
    explanation: str


class LogicEngine:
    def __init__(self):
        self.kb: List = []
        self.kb_strings: List[str] = []

    def load_kb_from_csv(self, csv_path: str) -> None:
        """
        Load FOL expressions from CSV file.
        Expected columns: id, fol
        """
        self.kb.clear()
        self.kb_strings.clear()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise RuntimeError("Task B KB CSV has no header row.")

            required_cols = {"id", "fol"}
            if not required_cols.issubset(set(reader.fieldnames)):
                raise RuntimeError("Task B KB CSV must contain headers: id,fol")

            for row in reader:
                fol_text = (row.get("fol") or "").strip()
                if fol_text:
                    self.kb.append(read_expr(fol_text))
                    self.kb_strings.append(fol_text)

        if not self.kb:
            raise RuntimeError("Task B KB loaded zero statements.")

    def kb_is_consistent(self) -> Tuple[bool, str]:
        """
        Conservative consistency check:
        For each statement S already in the KB, test whether NOT(S) is provable.
        If yes, KB contains a contradiction.
        """
        prover = ResolutionProver()

        for stmt in self.kb:
            neg_stmt = read_expr(f"-({stmt})")
            if prover.prove(neg_stmt, self.kb):
                return False, f"Contradiction found: NOT({stmt}) is provable."

        return True, "KB is consistent."

    def _prove(self, stmt_text: str) -> bool:
        prover = ResolutionProver()
        return prover.prove(read_expr(stmt_text), self.kb)

    def _has_explicit_fact(self, stmt_text: str) -> bool:
        return stmt_text in self.kb_strings

    def _pretty_entity(self, entity: str) -> str:
        return entity.replace("_", " ")
    
    
    def _split_predicate_and_entity(self, stmt_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Split a simple atomic fact like Vegan(butter) into:
        ("Vegan", "butter")
        Returns (None, None) if the format is not a simple atomic fact.
        """
        if "(" not in stmt_text or not stmt_text.endswith(")"):
            return None, None

        predicate, rest = stmt_text.split("(", 1)
        entity = rest[:-1].strip()  # remove closing ')'
        predicate = predicate.strip()

        if not predicate or not entity:
            return None, None

        return predicate, entity


    def _has_semantic_conflict(self, stmt_text: str) -> bool:
        """
        Lightweight domain-aware contradiction checks for common opposite concepts.
        This keeps the original formal negation check, but also blocks additions
        that clearly clash with food/allergy semantics in the coursework domain.
        """
        predicate, entity = self._split_predicate_and_entity(stmt_text)
        if not predicate or not entity:
            return False

        conflict_checks = {
            "Vegan": [
               f"Dairy({entity})",
               f"Egg({entity})"
               ],
            "DairyFree": [
                f"Dairy({entity})",
                f"NotDairyFree({entity})"
                ],
            "EggFree": [
                f"Egg({entity})",
                f"NotEggFree({entity})"
                ],
            "GlutenFree": [
                f"Gluten({entity})",
                f"NotGlutenFree({entity})"
                ],
            "NutFree": [
                f"Nut({entity})",
                f"NotNutFree({entity})"
                ],
            "SesameFree": [
                f"Sesame({entity})",
                f"NotSesameFree({entity})"
                ],
            "NotDairyFree": [
                f"DairyFree({entity})"
                ],
            "NotEggFree": [
                f"EggFree({entity})"
                ],
            "NotGlutenFree": [
                f"GlutenFree({entity})"
                ],
            "NotNutFree": [
                f"NutFree({entity})"
                ],
            "NotSesameFree": [
                f"SesameFree({entity})"
                ],
            }

        for conflicting_stmt in conflict_checks.get(predicate, []):
            if self._prove(conflicting_stmt):
                return True

        return False    
    
    
    

    def _build_positive_explanation(self, stmt_text: str) -> str:
        """
        Build a readable explanation for common Task B predicates.
        This is the Task B extension.
        """
        # Direct fact
        if self._has_explicit_fact(stmt_text):
            return f"The statement '{stmt_text}' is directly stored in the knowledge base."

        # UnsafeForMilkAllergy(x)
        if stmt_text.startswith("UnsafeForMilkAllergy("):
            entity = stmt_text[len("UnsafeForMilkAllergy("):-1]
            if self._prove(f"Dairy({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is unsafe for milk allergy because "
                    f"it is dairy, dairy items are not dairy-free, and items that are not dairy-free "
                    f"are unsafe for milk allergy."
                )
            if self._prove(f"NotDairyFree({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is unsafe for milk allergy because "
                    f"it is not dairy-free."
                )

        # UnsafeForCoeliac(x)
        if stmt_text.startswith("UnsafeForCoeliac("):
            entity = stmt_text[len("UnsafeForCoeliac("):-1]
            if self._prove(f"Gluten({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is unsafe for coeliac users because "
                    f"it contains gluten, gluten items are not gluten-free, and not gluten-free "
                    f"items are unsafe for coeliac users."
                )

        # UnsafeForEggAllergy(x)
        if stmt_text.startswith("UnsafeForEggAllergy("):
            entity = stmt_text[len("UnsafeForEggAllergy("):-1]
            if self._prove(f"Egg({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is unsafe for egg allergy because "
                    f"it contains egg, egg items are not egg-free, and not egg-free items are unsafe."
                )

        # UnsafeForNutAllergy(x)
        if stmt_text.startswith("UnsafeForNutAllergy("):
            entity = stmt_text[len("UnsafeForNutAllergy("):-1]
            if self._prove(f"Nut({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is unsafe for nut allergy because "
                    f"it contains nuts, nut items are not nut-free, and not nut-free items are unsafe."
                )

        # NotDairyFree(x)
        if stmt_text.startswith("NotDairyFree("):
            entity = stmt_text[len("NotDairyFree("):-1]
            if self._prove(f"Dairy({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is not dairy-free because it is classified as dairy."
                )

        # NotGlutenFree(x)
        if stmt_text.startswith("NotGlutenFree("):
            entity = stmt_text[len("NotGlutenFree("):-1]
            if self._prove(f"Gluten({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is not gluten-free because it contains gluten."
                )

        # NotEggFree(x)
        if stmt_text.startswith("NotEggFree("):
            entity = stmt_text[len("NotEggFree("):-1]
            if self._prove(f"Egg({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is not egg-free because it contains egg."
                )

        # NotNutFree(x)
        if stmt_text.startswith("NotNutFree("):
            entity = stmt_text[len("NotNutFree("):-1]
            if self._prove(f"Nut({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is not nut-free because it contains nuts."
                )

        # DairyFree(x)
        if stmt_text.startswith("DairyFree("):
            entity = stmt_text[len("DairyFree("):-1]
            if self._prove(f"Vegan({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is dairy-free because it is classified as vegan."
                )

        # EggFree(x)
        if stmt_text.startswith("EggFree("):
            entity = stmt_text[len("EggFree("):-1]
            if self._prove(f"Vegan({entity})"):
                return (
                    f"{self._pretty_entity(entity).title()} is egg-free because it is classified as vegan."
                )

        # Generic fallback
        return f"The statement '{stmt_text}' is provable from the knowledge base."

    def _build_negative_explanation(self, stmt_text: str) -> str:
        return f"The negation of '{stmt_text}' is provable from the knowledge base."

    def check_statement(self, stmt_text: str) -> LogicKBResult:
        """
        Return:
        - CORRECT   if statement is provable
        - INCORRECT if negation is provable
        - UNKNOWN   otherwise
        """
        prover = ResolutionProver()

        stmt = read_expr(stmt_text)
        neg_stmt = read_expr(f"-({stmt_text})")

        if prover.prove(stmt, self.kb):
            return LogicKBResult(
                "CORRECT",
                self._build_positive_explanation(stmt_text)
            )

        if prover.prove(neg_stmt, self.kb):
            return LogicKBResult(
                "INCORRECT",
                self._build_negative_explanation(stmt_text)
            )

        return LogicKBResult(
            "UNKNOWN",
            f"Neither '{stmt_text}' nor its negation can be proved from the knowledge base."
        )

    def add_statement_if_consistent(self, stmt_text: str) -> Tuple[bool, str]:
        """
        Add a new fact only if it does not contradict current knowledge.
        """
        prover = ResolutionProver()

        stmt = read_expr(stmt_text)
        neg_stmt = read_expr(f"-({stmt_text})")

        # Original formal contradiction check
        if prover.prove(neg_stmt, self.kb):
            return False, "That would contradict what I already know."

        # Extra domain-aware contradiction check
        if self._has_semantic_conflict(stmt_text):
            return False, "That would contradict what I already know."

        self.kb.append(stmt)
        self.kb_strings.append(stmt_text)
        return True, f"I have added '{stmt_text}' to the knowledge base."    