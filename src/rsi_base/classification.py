"""Evolvable memory retrieval, context selection and two-call classification harness."""

import math
import re
from collections import Counter
from typing import Literal

from pydantic import Field

from .types import AgentSpec, StrictModel


class MemoryPolicy(StrictModel):
    retrieval: Literal["none", "fixed", "bm25", "tfidf", "hybrid"] = "bm25"
    examples: int = Field(default=8, ge=0, le=32)
    selection: Literal["top", "diverse"] = "top"
    max_per_label: int = Field(default=32, ge=1, le=32)
    deduplicate: bool = True
    label_hints: bool = False
    order: Literal["best_first", "best_last"] = "best_first"
    workflow: Literal["direct", "verify", "rewrite"] = "direct"
    layout: Literal["examples", "coverage", "contrast_pairs"] = "examples"
    review_retrieval: Literal["same", "draft_contrast"] = "same"
    confirmers: int = Field(default=5, ge=1, le=16)
    challengers: int = Field(default=5, ge=1, le=16)
    system_prompt: str = Field(
        default="Classify the query using the allowed labels and relevant labeled examples. Distinguish similar classes using the full description.",
        min_length=1,
        max_length=1200,
    )


class MemoryCandidate(StrictModel):
    policy: MemoryPolicy
    reason: str = Field(min_length=1, max_length=1800)


class MemoryBatch(StrictModel):
    candidates: list[MemoryCandidate] = Field(min_length=2, max_length=2)


class SingleMemoryBatch(StrictModel):
    candidates: list[MemoryCandidate] = Field(min_length=1, max_length=1)


def effective_policy(policy):
    """Canonicalize unused knobs so ineffective edits cannot earn noisy 'improvements'."""
    values = policy.model_dump()
    if policy.workflow != "verify":
        values["review_retrieval"] = "same"
    if values["review_retrieval"] == "same":
        values.update(confirmers=5, challengers=5)
    if policy.retrieval != "none":
        if policy.layout == "contrast_pairs":
            values["max_per_label"] = 32
        elif policy.examples:
            values["max_per_label"] = min(policy.max_per_label, policy.examples)
    if policy.retrieval == "none":
        values.update(
            examples=0,
            selection="top",
            max_per_label=32,
            label_hints=False,
            order="best_first",
            layout="examples",
            review_retrieval="same",
            confirmers=5,
            challengers=5,
        )
    return MemoryPolicy.model_validate(values)


STOP = set(
    "a an and are as at be been but by can for from had has have i in is it its me my of on or that the their then there these they this to was were when with you your im ive feeling feel get getting really lot also lately".split()
)


def words(text):
    return [w for w in re.findall(r"[a-z]+", text.casefold()) if w not in STOP and len(w) > 1]


def grams(text):
    value = " ".join(words(text))
    return Counter(value[i : i + 3] for i in range(max(0, len(value) - 2)))


def normalize(text):
    return " ".join(text.casefold().split())


def label_score(response, reference, labels):
    prediction = normalize(response)
    valid = prediction in labels
    return {
        "parsed_answer": prediction,
        "score": int(valid and prediction == normalize(reference)),
        "status": "format_error"
        if not valid
        else "correct"
        if prediction == normalize(reference)
        else "wrong",
    }


class MemoryIndex:
    def __init__(self, training):
        self.training = tuple(dict(row) for row in training)

    def retrieve(self, query, policy, *, exclude_id=None):
        # Rebuild only from permitted memory rows; leave-one-out also excludes identical text.
        rows = [
            r
            for r in self.training
            if r["id"] != exclude_id and normalize(r["question"]) != normalize(query)
        ]
        if policy.retrieval == "none" or not policy.examples:
            return [], []
        tokens = [Counter(words(r["question"])) for r in rows]
        df = Counter(w for doc in tokens for w in doc)
        n = len(rows)
        avg = sum(sum(doc.values()) for doc in tokens) / max(1, n)
        q = Counter(words(query))
        query_grams = grams(query)
        ranks = []
        for index, (row, doc) in enumerate(zip(rows, tokens, strict=True)):
            size = sum(doc.values())
            bm25 = sum(
                math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                * doc[w]
                * 2.5
                / (doc[w] + 1.5 * (0.25 + 0.75 * size / max(1, avg)))
                for w in q
                if doc[w]
            )
            idf = lambda w: math.log((n + 1) / (df[w] + 1)) + 1  # noqa: E731
            norm = math.sqrt(sum((freq * idf(w)) ** 2 for w, freq in doc.items()))
            qnorm = math.sqrt(sum((freq * idf(w)) ** 2 for w, freq in q.items()))
            tfidf = sum(q[w] * doc[w] * idf(w) ** 2 for w in q) / max(1e-12, norm * qnorm)
            chars = grams(row["question"])
            charcos = sum(v * chars[k] for k, v in query_grams.items()) / max(
                1e-12,
                math.sqrt(
                    sum(v * v for v in query_grams.values()) * sum(v * v for v in chars.values())
                ),
            )
            value = {"fixed": -index, "bm25": bm25, "tfidf": tfidf, "hybrid": tfidf + charcos}[
                policy.retrieval
            ]
            ranks.append((value, row))
        ranks.sort(key=lambda x: (-x[0], x[1]["id"]))
        votes = Counter()
        for value, row in ranks[:16]:
            votes[row["answer"]] += max(value, 0)
        if policy.selection == "diverse":
            # First cover distinct labels, then permit additional evidence for each label.
            first, rest, seen_labels = [], [], set()
            for pair in ranks:
                label = pair[1]["answer"]
                (rest if label in seen_labels else first).append(pair)
                seen_labels.add(label)
            ranks = first + rest
        selected, counts, seen_text = [], Counter(), set()
        for value, row in ranks:
            text = normalize(row["question"])
            if (
                counts[row["answer"]] >= policy.max_per_label
                or policy.deduplicate
                and text in seen_text
            ):
                continue
            selected.append({**row, "retrieval_score": value})
            counts[row["answer"]] += 1
            seen_text.add(text)
            if len(selected) >= policy.examples:
                break
        if policy.order == "best_last":
            selected.reverse()
        return selected, votes.most_common(5)


class MemoryHarness:
    def __init__(self, training, labels, *, context_chars=12000):
        self.index, self.labels, self.context_chars = (
            MemoryIndex(training),
            tuple(labels),
            context_chars,
        )

    def render(self, question, blocks, *, hint=""):
        header = "Allowed labels: " + "; ".join(self.labels) + "\n\nLabeled training examples:\n"
        tail = (
            "\nQuery:\n"
            + question
            + "\n\nChoose the best label from Allowed labels. Output only its exact name, without brackets, tags, explanation or any other text."
        )
        if len(header + tail) > self.context_chars:
            raise ValueError("Query and label contract exceed fixed context budget")
        examples, used = "", []
        for title, rows in blocks:
            piece = (title + "\n" if title else "") + "".join(
                f"Description: {row['question']}\nLabel: {row['answer']}\n\n" for row in rows
            )
            # Keep a contrast pair intact rather than cutting off one side.
            if len(header + examples + piece + tail) <= self.context_chars:
                examples += piece
                used.extend(row["id"] for row in rows)
        if len(header + examples + hint + tail) > self.context_chars:
            hint = ""
        return header + examples + hint + tail, used

    def context(self, question, policy, *, query=None, exclude_id=None):
        query = query or question
        selected, hints = self.index.retrieve(query, policy, exclude_id=exclude_id)
        if policy.layout == "examples":
            blocks = [("", [row]) for row in selected]
        else:
            coverage, _ = self.index.retrieve(
                query,
                policy.model_copy(
                    update={"selection": "diverse", "examples": 32, "max_per_label": 1}
                ),
                exclude_id=exclude_id,
            )
            blocks = [("Label coverage example:", [row]) for row in coverage]
            if policy.layout == "coverage":
                blocks.extend(
                    ("Query-neighbor example:", [row])
                    for row in selected
                    if row["id"] not in {r["id"] for r in coverage}
                )
            else:
                neighbors, _ = self.index.retrieve(
                    query,
                    policy.model_copy(
                        update={"selection": "top", "examples": 32, "max_per_label": 32}
                    ),
                    exclude_id=exclude_id,
                )
                paired = set()
                for anchor in neighbors if policy.examples else []:
                    if anchor["id"] in paired:
                        continue
                    alternatives = [
                        row
                        for row in neighbors[:16]
                        if row["answer"] != anchor["answer"] and row["id"] not in paired
                    ]
                    if not alternatives:
                        continue
                    anchor_words = set(words(anchor["question"]))

                    def overlap(row, anchor_words=anchor_words):
                        tokens = set(words(row["question"]))
                        return len(tokens & anchor_words) / max(1, len(tokens | anchor_words)), row[
                            "retrieval_score"
                        ]

                    challenger = max(alternatives, key=overlap)
                    paired.update([anchor["id"], challenger["id"]])
                    blocks.append(
                        (
                            "Contrast: similar descriptions can have different labels. Compare distinguishing details:",
                            [anchor, challenger],
                        )
                    )
                    if len(paired) >= policy.examples:
                        break
        hint = (
            "\nRetrieval suggestions (noisy; verify against the query): "
            + "; ".join(k for k, _ in hints)
            if policy.label_hints
            else ""
        )
        return self.render(question, blocks, hint=hint)

    def contrast_review(self, question, draft, policy, *, exclude_id=None):
        label = normalize(draft)
        if label not in self.labels:
            return self.context(question, policy, exclude_id=exclude_id)
        confirm_index = MemoryIndex([r for r in self.index.training if r["answer"] == label])
        challenge_index = MemoryIndex([r for r in self.index.training if r["answer"] != label])
        confirm, _ = confirm_index.retrieve(
            question,
            policy.model_copy(
                update={"examples": policy.confirmers, "selection": "top", "max_per_label": 32}
            ),
            exclude_id=exclude_id,
        )
        challenge, _ = challenge_index.retrieve(
            question,
            policy.model_copy(
                update={"examples": policy.challengers, "selection": "top", "max_per_label": 32}
            ),
            exclude_id=exclude_id,
        )
        blocks = [
            ("Retrieved examples with the draft label (not proof that the draft is correct):", [r])
            for r in confirm
        ]
        blocks += [("Retrieved counterexamples with DIFFERENT labels:", [r]) for r in challenge]
        return self.render(question, blocks)

    def run(self, policy, problem, call, *, leave_one_out=False):
        question, query, traces = problem["question"], None, []
        spec = AgentSpec(system_prompt=policy.system_prompt)
        if policy.workflow == "rewrite":
            result = call(
                "rewrite",
                AgentSpec(
                    system_prompt="Extract concise, discriminative search terms from the query. Preserve negations and details. Do not answer or invent information."
                ),
                question,
            )
            traces.append(result)
            query = question + "\n" + result["response"][:1500]
        prompt, used = self.context(
            question, policy, query=query, exclude_id=problem["id"] if leave_one_out else None
        )
        result = call("predict", spec, prompt)
        traces.append(result)
        memory_by_stage = {"predict": used}
        if policy.workflow == "verify":
            if policy.review_retrieval == "draft_contrast":
                prompt, review_used = self.contrast_review(
                    question,
                    result["response"],
                    policy,
                    exclude_id=problem["id"] if leave_one_out else None,
                )
            else:
                review_used = used
            memory_by_stage["verify"] = review_used
            used = list(dict.fromkeys(used + review_used))
            review = (
                prompt
                + "\n\nDraft label (may be wrong):\n"
                + result["response"][:1200]
                + "\nIndependently compare plausible labels, check distinguishing details, and return the best label in the required format."
            )
            if len(review) > self.context_chars + 1500:
                raise ValueError("Review exceeds fixed context allowance")
            result = call("verify", spec, review)
            traces.append(result)
        return {
            "response": result["response"],
            "memory_ids": used,
            "memory_by_stage": memory_by_stage,
            "traces": traces,
            "model_calls": len(traces),
        }
