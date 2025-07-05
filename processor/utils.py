
TIME_SPLIT = [
    (("2020-12-01", "2023-01-01"), "pretrain"),
    (("2023-01-01", "2023-02-01"), "phase1"),
    (("2023-02-01", "2023-03-01"), "phase2"),
    (("2023-03-01", "2023-04-01"), "test"),
]

def time_split_data(df, time_split=TIME_SPLIT):
    results = {}
    for (start, end), phase in time_split:
        df_time = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
        results[phase] = df_time
    return results
    
