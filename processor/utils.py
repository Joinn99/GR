
TIME_SPLIT = [
    (("2017-07-01", "2022-07-01"), "pretrain"),
    (("2022-07-01", "2022-10-01"), "phase1"),
    (("2022-10-01", "2023-01-01"), "phase2"),
    (("2023-01-01", "2023-04-01"), "test"),
]

def time_split_data(df, time_split=TIME_SPLIT):
    results = {}
    for (start, end), phase in time_split:
        df_time = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
        results[phase] = df_time
    return results
    
