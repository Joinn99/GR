from logger import get_logger, log_with_color

logger = get_logger(__name__)

TIME_SPLIT = [
    (("2017-07-01", "2022-07-01"), "pretrain"),
    (("2022-07-01", "2022-10-01"), "phase1"),
    (("2022-10-01", "2023-01-01"), "phase2"),
    (("2023-01-01", "2023-04-01"), "test"),
]

def time_split_data(df, time_split=TIME_SPLIT):
    log_with_color(logger, "INFO", f"Splitting data with {len(df)} records into {len(time_split)} time periods", "red")
    results = {}
    for (start, end), phase in time_split:
        df_time = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
        results[phase] = df_time
        log_with_color(logger, "INFO", f"Phase {phase}: {len(df_time)} records ({start} to {end})", "red")
    return results
    
