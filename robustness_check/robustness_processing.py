"""
Data processing function for robustness checks.
This function encapsulates the EDA pipeline with configurable parameters.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def process_data_for_robustness(
    early_window_size=3,
    label_threshold=4,
    last_action_day_threshold=10
):
    """
    Process data with configurable parameters for robustness checks.
    
    Parameters:
    -----------
    early_window_size : int
        Number of early actions to consider (2, 3, or 5)
    label_threshold : float
        Weighted score threshold for labeling active users
    last_action_day_threshold : int
        Minimum last action day for active users
    
    Returns:
    --------
    df_user : pd.DataFrame
        User-level dataset ready for modeling
    """
    
    print(f"\n{'='*60}")
    print(f"Processing with: early_window={early_window_size}, "
          f"label_threshold={label_threshold}, last_action_day={last_action_day_threshold}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv("../data/users_with_actions.csv")
    if "registeredMonthCnt" in df.columns:
        df.drop("registeredMonthCnt", axis=1, inplace=True)
    
    # Define action columns
    all_actions = [
        'isClick', 'isComment', 'isIntoPersonalHomepage', 'isShare',
        'isViewComment', 'isLike', 'mlogViewTime'
    ]
    
    # Count actions per impression and cumulative per user
    df['num_actions_row'] = df[all_actions].sum(axis=1)
    df = df.sort_values(['userId', 'dt', df.index.name or 'mlogId'])
    df['cum_actions'] = df.groupby('userId')['num_actions_row'].cumsum()
    
    # Rows strictly AFTER the first N accumulated actions
    df_after = df[df['cum_actions'] > early_window_size].copy()
    
    # Weighted engagement
    df_after['weighted_score_row'] = (
        1 * df_after['isClick'] +
        3 * df_after['isLike'] +
        4 * df_after['isComment'] +
        4 * df_after['isShare'] +
        2 * df_after['isViewComment'] +
        2 * df_after['isIntoPersonalHomepage'] +
        0.01 * df_after['mlogViewTime']
    )
    
    user_weighted = df_after.groupby('userId')['weighted_score_row'].sum()
    y_active_weighted = (user_weighted >= label_threshold).astype(int).rename('y_active_weighted')
    
    # Last action day
    df['any_action'] = (df[all_actions].sum(axis=1) > 0).astype(int)
    last_action_dt = (
        df[df['any_action'] == 1]
        .groupby('userId')['dt']
        .max()
        .rename('last_action_dt')
    )
    last_action_dt = last_action_dt.reindex(df['userId'].unique(), fill_value=0)
    
    # Combine user-level labels
    user_labels = pd.concat([y_active_weighted, last_action_dt, user_weighted], axis=1)
    user_labels = user_labels.fillna(0)
    user_labels['y_active_weighted'] = user_labels['y_active_weighted'].astype(int)
    user_labels['last_action_dt'] = user_labels['last_action_dt'].astype(int)
    user_labels['weighted_score_row'] = user_labels['weighted_score_row'].astype(float)
    
    # Final user-level label with configurable threshold
    user_labels['y_active'] = (
        (user_labels['y_active_weighted'] == 1) &
        (user_labels['last_action_dt'] >= last_action_day_threshold)
    ).astype(int)
    
    # Map label back to full df
    df = df.drop(columns=['y_active'], errors='ignore')
    df = df.merge(
        user_labels[['y_active', 'weighted_score_row']],
        left_on='userId',
        right_index=True,
        how='left'
    )
    df[['y_active']] = df[['y_active']].fillna(0).astype(int)
    
    # Clean up helper columns
    df.drop(columns=['num_actions_row', 'cum_actions', 'any_action'],
            inplace=True, errors='ignore')
    
    # Temporal features
    df["day"] = pd.to_datetime("2019-11-01") + pd.to_timedelta(df["dt"] - 1, unit="D")
    df['isWeekend'] = (df['day'].dt.dayofweek >= 5).astype(int)
    
    # Missing data handling
    def to_na(s: pd.Series):
        return (s.astype(str).str.strip()
                .replace({"": np.nan, "nan": np.nan, "None": np.nan,
                          "unknown": np.nan, "UNKNOWN": np.nan}))
    
    if "age" in df.columns:
        df["age_missing"] = df["age"].isna().astype("uint8")
        df = df.drop(columns=["age"])
    if "gender" in df.columns:
        df = df.drop(columns=["gender"])
    
    if "age_missing" in df.columns:
        df.rename(columns={"age_missing": "age_gender_missing"}, inplace=True)
    else:
        df["age_gender_missing"] = pd.Series(0, index=df.index, dtype="uint8")
    
    if "contentId" in df.columns:
        df["contentId"] = to_na(df["contentId"]).fillna("unknown").astype("string")
    
    if "mlogViewTime" in df.columns:
        df["mlogViewTime"].fillna(0, inplace=True)
    
    df = df.dropna(subset=["creatorId"]).reset_index(drop=True)
    df = df[df["creatorId"].notna()].reset_index(drop=True)
    
    if "impressTime" in df.columns:
        df.drop("impressTime", axis=1, inplace=True)
    
    # Popularity features with PCA
    pop_cols = [
        "userImprssionCount", "userClickCount", "userLikeCount", "userCommentCount",
        "userShareCount", "userViewCommentCount", "userIntoPersonalHomepageCount",
        "userFollowCreatorCount"
    ]
    
    for c in pop_cols:
        if c in df.columns:
            df[c+"_log"] = np.log1p(df[c].astype("float32"))
            df[c+"_log_zdt"] = df.groupby("dt")[c+"_log"].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-6)
            ).astype("float32")
    
    z_cols = [c for c in df.columns if c.endswith("_log_zdt")]
    if z_cols:
        pca = PCA(n_components=1, random_state=42)
        df["pop_index_pca"] = pca.fit_transform(df[z_cols].fillna(0))[:, 0].astype("float32")
        
        pop_related_cols = [c for c in df.columns if any(term in c for term in pop_cols)]
        cols_to_drop = [c for c in pop_related_cols if c != "pop_index_pca"]
        df = df.drop(columns=cols_to_drop, errors="ignore")
    
    # Additional popularity indices
    if 'contentId' in df.columns:
        content_pop_index = (
            df.groupby('contentId')['pop_index_pca']
            .mean()
            .rename('content_pop_index')
        )
        df = df.merge(content_pop_index, on='contentId', how='left')
    
    if 'creatorId' in df.columns:
        creator_pop_index = (
            df.groupby('creatorId')['pop_index_pca']
            .mean()
            .rename('creator_pop_index')
        )
        df = df.merge(creator_pop_index, on='creatorId', how='left')
    
    df['position_inv'] = 1 / (df['impressPosition'] + 1e-6)
    df['pop_rank_exposure_row'] = np.log1p(np.abs(
        df['position_inv'] * df['pop_index_pca']
    )).astype('float64')
    
    # Transform variables
    for col in ["mlogViewTime", "followCnt", "impressPosition", "publishTime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "publishTime" in df.columns:
        cap_pub = df["publishTime"].quantile(0.995)
        if pd.notna(cap_pub):
            df.loc[df["publishTime"] > cap_pub, "publishTime"] = cap_pub
    
    if "mlogViewTime" in df.columns:
        vt = df["mlogViewTime"].fillna(0).clip(lower=0).astype("float64")
        pos = vt > 0
        q_hi = vt.loc[pos].quantile(0.99) if pos.sum() > 0 else 0
        if pd.notna(q_hi) and q_hi > 0:
            vt.loc[pos & (vt > q_hi)] = float(q_hi)
        df["mlogViewTime_log"] = np.log1p(vt)
    
    if "impressPosition" in df.columns:
        df.loc[df["impressPosition"] == 0, "impressPosition"] = np.nan
        imp_med_day = df.groupby("day", observed=True)["impressPosition"].transform("median") if "day" in df.columns else np.nan
        imp_med_global = df["impressPosition"].median(skipna=True)
        df["impressPosition"] = df["impressPosition"].fillna(imp_med_day).fillna(imp_med_global)
        df["impressPosition_log"] = np.log1p(df["impressPosition"])
    
    if "followCnt" in df.columns:
        df["followCnt"] = pd.to_numeric(df["followCnt"], errors="coerce")
        df["followCnt_log"] = np.log1p(df["followCnt"].fillna(0).clip(lower=0))
    
    # Filtering early actions (using early_window_size)
    first_action_dt = (
        df.groupby('userId')['dt']
        .min()
        .rename('first_action_dt')
    )
    df = df.merge(first_action_dt, left_on='userId', right_index=True, how='left')
    df['first_action_dt'] = df['first_action_dt'].fillna(0).astype(int)
    
    excluded_users = df.loc[df['first_action_dt'] >= 25, 'userId'].unique()
    df = df[~df['userId'].isin(excluded_users)].copy()
    
    # Filter to first N impressions per user (using early_window_size)
    df = (
        df.sort_values(['userId', 'dt'])
        .groupby('userId')
        .head(early_window_size)
        .reset_index(drop=True)
    )
    
    df.drop(columns=['first_action_dt'], inplace=True, errors='ignore')
    
    # Feature Engineering
    acts = ['isClick', 'isComment', 'isIntoPersonalHomepage', 'isShare', 'isViewComment', 'isLike']
    for c in acts:
        if c in df.columns:
            df[c] = df[c].astype(int)
    
    grp = df.groupby('userId', as_index=True)
    
    agg = grp.agg(
        actions_taken=('mlogId', 'count'),
        clicks=('isClick', 'sum'),
        likes=('isLike', 'sum'),
        comments=('isComment', 'sum'),
        shares=('isShare', 'sum'),
        homeviews=('isIntoPersonalHomepage', 'sum'),
        viewcoms=('isViewComment', 'sum'),
        followCnt_log=('followCnt_log', 'max')
    )
    
    den_imp = agg['actions_taken'].clip(lower=1)
    agg['like_rate'] = agg['likes'] / den_imp
    agg['comment_rate'] = agg['comments'] / den_imp
    agg['share_rate'] = agg['shares'] / den_imp
    
    agg['ever_shared'] = (agg['shares'] > 0).astype(int)
    agg['ever_commented'] = (agg['comments'] > 0).astype(int)
    agg['ever_inthomeviewed'] = (agg['homeviews'] > 0).astype(int)
    
    action_sums = agg[['clicks', 'likes', 'comments', 'shares', 'homeviews', 'viewcoms']]
    counts = action_sums.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    p = np.divide(counts, np.where(row_sums==0, 1, row_sums), where=(row_sums!=0))
    entropy = -np.nansum(np.where(p>0, p*np.log(p), 0.0), axis=1)
    entropy[row_sums.flatten()==0] = 0.0
    agg['action_entropy'] = entropy
    
    agg['shares_per_exp_follow'] = agg['shares'] / np.exp(agg['followCnt_log']).clip(lower=1)
    
    for a in ['isLike', 'isComment']:
        name = f'sum_{a}_x_followlog'
        agg[name] = df.groupby('userId').apply(
            lambda g: (g[a] * g['followCnt_log']).sum(), include_groups=False
        )
    
    agg['action_rate'] = (agg['clicks'] + agg['likes'] + agg['comments'] + agg['shares']) / den_imp
    agg['followlog_x_actionrate'] = agg['followCnt_log'] * agg['action_rate']
    
    def _shannon_entropy(vals: pd.Series) -> float:
        p = vals.value_counts(normalize=True, dropna=True).to_numpy(dtype=float)
        if p.size == 0:
            return 0.0
        return float(-(p * np.log(p)).sum())
    
    content_entropy = (
        df.groupby('userId')['contentId'].apply(_shannon_entropy).rename('content_entropy_seen')
    )
    talk_entropy = (
        df.groupby('userId')['talkId'].apply(_shannon_entropy).rename('talk_entropy_seen')
    )
    
    span = (
        df.groupby('userId')['dt'].agg(lambda s: int(s.max() - s.min())).rename('engagement_span')
    )
    
    avg_dwell = (
        df.groupby('userId')['mlogViewTime_log'].mean().rename('avg_dwell')
    )
    
    avg_position = (
        df.groupby('userId')['impressPosition_log'].mean().rename('avg_position')
    )
    
    active_days = (
        df.groupby('userId')['dt']
        .nunique()
        .rename('active_days')
    )
    
    weekend_share = (
        df.groupby('userId')['isWeekend']
        .mean()
        .rename('weekend_share')
    )
    
    only_click = df.groupby('userId').apply(
        lambda g: int((g['isClick'].sum() > 0) and 
                     (g[['isLike', 'isComment', 'isShare', 'isViewComment', 'isIntoPersonalHomepage']].sum().sum() == 0)),
        include_groups=False
    ).rename('only_click')
    
    shared_x_liked = df.groupby('userId').apply(
        lambda g: int(((g['isShare'] == 1) & (g['isLike'] == 1)).any()),
        include_groups=False
    ).rename('shared_x_liked')
    
    shared_x_commented = df.groupby('userId').apply(
        lambda g: int(((g['isShare'] == 1) & (g['isComment'] == 1)).any()),
        include_groups=False
    ).rename('shared_x_commented')
    
    into_home_x_liked = df.groupby('userId').apply(
        lambda g: int(((g['isIntoPersonalHomepage'] == 1) & (g['isLike'] == 1)).any()),
        include_groups=False
    ).rename('intohome_x_liked')
    
    agg = agg.join([content_entropy, talk_entropy, span, avg_dwell, avg_position, 
                    active_days, weekend_share, only_click, shared_x_liked, 
                    shared_x_commented, into_home_x_liked], how='left')
    
    agg = agg.fillna({'content_entropy_seen': 0, 'talk_entropy_seen': 0, 'engagement_span': 0,
                     'avg_dwell': 0.0, 'avg_position': 0.0, 'active_days': 0, 'weekend_share': 0,
                     'only_click': 0, 'shared_x_liked': 0, 'shared_x_commented': 0, 'intohome_x_liked': 0})
    
    agg = agg.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    features_per_user = agg[[
        'actions_taken', 'like_rate', 'share_rate', 'comment_rate', 'ever_inthomeviewed',
        'ever_shared', 'ever_commented', 'action_entropy', 'shares_per_exp_follow', 'action_rate',
        'sum_isLike_x_followlog', 'sum_isComment_x_followlog', 'followlog_x_actionrate',
        'content_entropy_seen', 'talk_entropy_seen', 'engagement_span', 'avg_dwell', 'avg_position',
        'active_days', 'weekend_share', 'only_click', 'shared_x_liked', 'shared_x_commented', 'intohome_x_liked'
    ]]
    
    df = df.merge(
        features_per_user.add_prefix("usr_"),
        how="left",
        left_on="userId",
        right_index=True
    )
    
    # Create user-level dataset
    usr_cols = [c for c in df.columns if c.startswith('usr_')]
    
    cont_avg_cols = [
        'pop_index_pca', 'content_pop_index', 'pop_rank_exposure_row', 'publishTime',
    ]
    
    cont_max_cols = ['creator_pop_index']
    
    agg_dict = {
        'y_active': ('y_active', 'first'),
        'province': ('province', 'first'),
        'age_gender_missing': ('age_gender_missing', 'first'),
        'followCnt_log': ('followCnt_log', 'max'),
    }
    
    agg_dict.update({col: (col, 'first') for col in usr_cols})
    agg_dict.update({f'avg_{col}': (col, 'mean') for col in cont_avg_cols if col in df.columns})
    agg_dict.update({f'max_{col}': (col, 'max') for col in cont_max_cols if col in df.columns})
    
    df_user = df.groupby('userId', as_index=False).agg(**agg_dict)
    
    # PCA for entropy features
    if all(c in df_user.columns for c in ['usr_content_entropy_seen', 'usr_talk_entropy_seen', 'usr_actions_taken']):
        df_e = df_user[['usr_content_entropy_seen', 'usr_talk_entropy_seen', 'usr_actions_taken']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_e)
        pca = PCA(n_components=1, random_state=42)
        df_user['entropy_pca'] = pca.fit_transform(X_scaled)
        df_user.drop(
            columns=['usr_content_entropy_seen', 'usr_talk_entropy_seen', 'usr_action_entropy', 'usr_actions_taken'],
            inplace=True,
            errors='ignore'
        )
    
    # Winsorize heavy tails
    heavy_tail_cols = [
        'followCnt_log', 'usr_sum_isLike_x_followlog', 'usr_sum_isComment_x_followlog',
        'usr_followlog_x_actionrate', 'usr_shares_per_exp_follow', 'avg_pop_rank_exposure_row',
    ]
    
    upper_q = 0.995
    for col in heavy_tail_cols:
        if col in df_user.columns:
            upper = df_user[col].quantile(upper_q)
            if pd.notna(upper):
                df_user[col] = np.minimum(df_user[col], upper)
    
    print(f"Final user-level dataset shape: {df_user.shape}")
    print(f"Target distribution:\n{df_user['y_active'].value_counts(normalize=True)}")
    
    return df_user

