"""
ChemGIME EC Prediction
======================
Enzyme Commission (EC) number prediction via similarity-weighted voting.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _calculate_prediction_from_scores(scores_dict: dict, level_name: str) -> tuple[str, float, pd.DataFrame]:
    """Helper to calculate winner, confidence, and score DF."""
    col_names = [level_name, "Confidence"]

    if not scores_dict:
        return "N/A", 0.0, pd.DataFrame(columns=col_names)

    total_score = sum(scores_dict.values())
    if total_score == 0:
        return "N/A", 0.0, pd.DataFrame(columns=col_names)

    predicted_ec = max(scores_dict, key=scores_dict.get)
    confidence = scores_dict[predicted_ec] / total_score

    sorted_scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    score_data = {
        col_names[0]: [ec for ec, score in sorted_scores],
        col_names[1]: [f"{(score / total_score):.2%}" for ec, score in sorted_scores]
    }
    scores_df = pd.DataFrame(score_data)

    return predicted_ec, confidence, scores_df.head(10)


def predict_ec_weighted(df_rank, distance_threshold=0.5):
    """Predicts EC numbers for all 4 levels using a similarity-weighted vote."""
    ec_l1_scores, ec_l2_scores, ec_l3_scores, ec_l4_scores = {}, {}, {}, {}
    df_vote = df_rank[df_rank['tan_distance'] <= distance_threshold].copy()

    def get_empty_result(level_name):
        return "N/A", 0.0, pd.DataFrame(columns=[level_name, "Confidence"])
    empty_results = {
        "L1": get_empty_result("EC Class (L1)"),
        "L2": get_empty_result("EC Subclass (L2)"),
        "L3": get_empty_result("EC Sub-subclass (L3)"),
        "L4": get_empty_result("EC Serial (L4)")
    }

    if df_vote.empty:
        logger.warning(f"No reactions found within distance threshold {distance_threshold} for EC prediction.")
        return empty_results

    has_valid_ec = False
    for _, row in df_vote.iterrows():
        distance = row['tan_distance']
        ec_string = str(row.get('ec', ''))
        if not ec_string or ec_string == 'nan' or ec_string == '-':
            continue

        vote_weight = (1.0 - distance)**2

        for ec in ec_string.split(';'):
            ec_clean = ec.strip()
            if not ec_clean:
                continue
            parts = ec_clean.split('.')

            if len(parts) >= 1 and parts[0].isdigit():
                ec_key = f"{parts[0]}.-.-.-"
                ec_l1_scores[ec_key] = ec_l1_scores.get(ec_key, 0.0) + vote_weight
                has_valid_ec = True
            if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
                ec_key = f"{parts[0]}.{parts[1]}.-.-"
                ec_l2_scores[ec_key] = ec_l2_scores.get(ec_key, 0.0) + vote_weight
            if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
                ec_key = f"{parts[0]}.{parts[1]}.{parts[2]}.-"
                ec_l3_scores[ec_key] = ec_l3_scores.get(ec_key, 0.0) + vote_weight
            if len(parts) >= 4 and all(p.isdigit() for p in parts[:4]):
                ec_key = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"
                ec_l4_scores[ec_key] = ec_l4_scores.get(ec_key, 0.0) + vote_weight

    if not has_valid_ec:
        logger.warning("Reactions found, but none had valid EC numbers for prediction.")
        return empty_results

    results = {
        "L1": _calculate_prediction_from_scores(ec_l1_scores, "EC Class (L1)"),
        "L2": _calculate_prediction_from_scores(ec_l2_scores, "EC Subclass (L2)"),
        "L3": _calculate_prediction_from_scores(ec_l3_scores, "EC Sub-subclass (L3)"),
        "L4": _calculate_prediction_from_scores(ec_l4_scores, "EC Serial (L4)"),
    }

    logger.info(f"EC L3 Prediction: {results['L3'][0]} with {results['L3'][1]:.2%} confidence.")
    return results
