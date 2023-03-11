from collections import defaultdict

import pandas as pd


def precision_recall_at_k(predictions, ground_truth, k=10 ):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    # user_ = defaultdict(list)
    # for uid, true_r in zip(ground_truth.iterrows(), predictions.iterrows()):
    #
    #     user_est_true[uid].append((est, true_r))
    #
    precisions = dict()
    recalls = dict()
    for user_id, user_items_gt in ground_truth.groupby('users')['items']:

        predicted_items = predictions[predictions['users'] == user_id]['items']

        # Number of relevant items
        n_rel = len(user_items_gt)

        # Number of recommended items in top k
        n_rec_k = len(predicted_items[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = len(set(predicted_items[:k].values).intersection(set(user_items_gt.values)))

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


if __name__ == '__main__':
    from datetime import timedelta
    from src.data.ttrs_utils import create_user_x_items
    import numpy as np

    ttrs_dataset = pd.read_csv('data/processed/TTRS_dataset/data/tinkoff_raw_data_anonymized.csv')
    last_month = pd.to_datetime(ttrs_dataset['transaction_dttm'].max()) - timedelta(days=30)

    ttrs_dataset_last_month = ttrs_dataset[pd.to_datetime(ttrs_dataset['transaction_dttm']) >= last_month]
    ttrs_dataset_historical = ttrs_dataset[pd.to_datetime(ttrs_dataset['transaction_dttm']) < last_month]

    ttrs_dataset_last_month = create_user_x_items(ttrs_dataset_last_month)

    precisions, recalls = precision_recall_at_k(ttrs_dataset_last_month, ttrs_dataset_last_month)

    mean_precisions = np.mean(list(precisions.values()))
    mean_recalls = np.mean(list(recalls.values()))
    print(mean_precisions, mean_recalls)
