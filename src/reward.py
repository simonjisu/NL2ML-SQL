# EVAL

import re
from difflib import SequenceMatcher
from loguru import logger


class RewardCalculator:
    def __init__(
        self,
        weights = None,
        use_fuzzy_matching = True,
        normalize = True,  # True means `filter` mode other than `grpo` mode
        verbose = False
    ):
        self.use_fuzzy_matching = use_fuzzy_matching
        self.verbose = verbose
        self.normalize = normalize
        self.fail_val = 0.0 if normalize else -1.0

        # Fixed weights
        self.weights = weights or {
            "time_series": 0.1,
            "target_column": 0.5,
            "inference_condition": 0.3,
            "update_condition": 0.4,
            "task": 0.7
        }

        # Required keys
        self.required_keys = ["time_series", "target_column", "inference_condition", "task"]


    def _match(self, a, b, key=None):

        def jaccard(set1, set2):
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            if union != 0:
                # Jaccard similarity
                result = intersection / union
                if result == 0.0:
                    return self.fail_val
                return result
            else:
                return self.fail_val

        def extract_condition_parts(condition_str):
            condition_str = condition_str.strip()

            col = re.findall(r"<col>(.*?)</col>", condition_str)
            op = re.findall(r"<op>(.*?)</op>", condition_str)
            val = re.findall(r"<val>(.*?)</val>", condition_str)

            col_val = col[0].strip() if col else ""
            op_val = op[0].strip() if op else ""
            val_val = val[0].strip() if val else ""

            # Step 2: Heuristics if any of the parts are missing
            if not (col_val and op_val and val_val):
                # Remove tags to simplify raw parsing
                clean_str = re.sub(r"</?[^>]+>", "", condition_str)

                # Try simple expression pattern: col op val
                match = re.match(r"([a-zA-Z0-9_.]+)\s*([=!<>]+)\s*(.+)", clean_str)
                if match:
                    if not col_val:
                        col_val = match.group(1).strip()
                    if not op_val:
                        op_val = match.group(2).strip()
                    if not val_val:
                        val_val = match.group(3).strip()

            return col_val, op_val, val_val

        def tag_completeness_score(cond_str, tags):
            present = sum(tag in cond_str for tag in tags)
            return present / len(tags)
        
        def score_pair(a_cond, b_cond):
            a_cond = a_cond.strip().lower()
            b_cond = b_cond.strip().lower()
            a_col, a_op, a_val = extract_condition_parts(a_cond)
            b_col, b_op, b_val = extract_condition_parts(b_cond)

            col_score = jaccard({a_col}, {b_col})
            op_score = jaccard({a_op}, {b_op})
            val_score = int(SequenceMatcher(None, a_val, b_val).ratio() >= 0.9)

            if col_score == 0 or op_score == 0 or val_score == 0:
                return self.fail_val

            avg_score = (col_score + op_score + val_score) / 3
            tag_score = tag_completeness_score(a_cond, ["<col>", "</col>", "<op>", "</op>", "<val>", "</val>"])
            if tag_score != 1.0:
                return self.fail_val

            return avg_score

        if key in {"inference_condition", "update_condition"}:
            a_list = a if isinstance(a, list) else [a]
            b_list = b if isinstance(b, list) else [b]

            if not a_list and not b_list:
                return 1.0
            if not a_list or not b_list:
                return self.fail_val

            used_b_indices = set()
            matched_scores = []

            for a_cond in a_list:
                best_score = self.fail_val
                best_j = None
                for j, b_cond in enumerate(b_list):
                    if j in used_b_indices:
                        continue
                    score = score_pair(a_cond, b_cond)
                    if score > best_score:
                        best_score = score
                        best_j = j
                if best_j is not None:
                    used_b_indices.add(best_j)
                matched_scores.append(best_score)  # score is 0.0 if unmatched

            # Final score is average over max(len(predicted), len(ground_truth))
            final_score = sum(matched_scores) / max(len(a_list), len(b_list))
            return final_score

        # Default Jaccard (for all non-condition fields)
        a_str = " ".join(map(str, a)) if isinstance(a, list) else str(a)
        b_str = " ".join(map(str, b)) if isinstance(b, list) else str(b)

        if not a_str.strip() and not b_str.strip():
            return 1.0
        if not a_str.strip() or not b_str.strip():
            return self.fail_val

        if key == "target_column":

            def strip_tags(text):
                return re.sub(r"</?[^>]+>", "", text).strip().lower()

            a_clean = strip_tags(a_str)
            b_clean = strip_tags(b_str)

            tag_score = tag_completeness_score(a_str, ["<col>", "</col>"])
            if tag_score != 1.0:
                return self.fail_val

            sim_score = jaccard({a_clean}, {b_clean})
            return sim_score

        a_tokens = set(a_str.lower().split())
        b_tokens = set(b_str.lower().split())

        return jaccard(a_tokens, b_tokens)


    def weighted_accuracy(self, predicted, ground_truth):
        """Computes weighted accuracy + diagnostics"""
        matches = {}
        diagnostics = {}

        weights = self.weights.copy()

        # Required keys
        for key in self.required_keys:
            sim_score = self._match(predicted.get(key, []), ground_truth.get(key, []), key=key)
            matches[key] = sim_score
            diagnostics[key] = weights.get(key, 0) * sim_score

        # Optional key: update_condition
        has_update_condition = "update_condition" in ground_truth and (ground_truth.get("update_condition") not in (None, []))
        if has_update_condition:
            if predicted.get("update_condition") is None:
                predicted['update_condition'] = []
            sim_score = self._match(
                predicted.get("update_condition", []),
                ground_truth.get("update_condition", []),
                key="update_condition"
            )
            matches["update_condition"] = sim_score
            diagnostics["update_condition"] = weights.get("update_condition") * sim_score

        # Dynamically decide which keys to include in normalization
        active_keys = self.required_keys.copy()
        if has_update_condition:
            active_keys.append("update_condition")
        
        if self.normalize:
            # filter mode: normalize the score
            max_possible_score = sum(weights.get(k, 0) for k in active_keys)
            weighted_score = sum(diagnostics.values())
            final_score = max(0.0, min(1.0, weighted_score / max_possible_score))
        else:
            # grpo mode: No normalization
            final_score = sum(diagnostics.values())

        if self.verbose:
            logger.info("[Reward Diagnostics]")
            logger.info("Matches:", matches)
            logger.info("Diagnostics (per-key contribution):", diagnostics)
            logger.info("Weighted Score:", final_score)

        return round(final_score, 6), matches, diagnostics

    def get_min_possible_reward(self, ground_truth, convert_to_max=False):
        w = sum([self.weights.get(k) for k in ground_truth.keys() if self.weights.get(k) is not None])
        r = round(self.fail_val * w, 6)
        return -r if convert_to_max else r

    def self_check(self, intermediate_output):
        """Checks for presence of required keys."""
        for key in self.required_keys:
            if key not in intermediate_output:
                return {"status": "INVALID", "reason": f"Missing required key: {key}"}
        return {"status": "VALID", "intermediate_output": intermediate_output}