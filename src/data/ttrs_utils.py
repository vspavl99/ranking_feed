import pandas as pd


def create_user_x_items(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Extract from dataframe of transactions table of user_x_items
    :param dataframe: raw dataframe of transactions
    :return: user_x_items dataframe
    """
    user_x_items = dataframe.groupby(['party_rk', 'merchant_group_rk'])['category'].count().reset_index()
    user_x_items.columns = ['users', 'items', 'counts']

    return user_x_items


