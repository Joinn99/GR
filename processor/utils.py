
TIME_SPLIT = [
    (("2012-12-01", "2014-02-01"), "pretrain"),
    (("2014-02-01", "2014-04-01"), "phase1"),
    (("2014-04-01", "2014-06-01"), "phase2"),
    (("2014-06-01", "2014-08-01"), "test"),
]

def time_split_data(df, time_split=TIME_SPLIT):
    results = {}
    for (start, end), phase in time_split:
        df_time = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
        results[phase] = df_time
    return results
    
