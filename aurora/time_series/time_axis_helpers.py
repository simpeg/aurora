import numpy as np
import pandas as pd
import time


def fast_arange(t0, n_samples, sample_rate):
    t0 = np.datetime64(t0)
    dt = 1.0 / sample_rate
    dt_nanoseconds = int(np.round(1e9 * dt))
    dt_timedelta = np.timedelta64(dt_nanoseconds, "ns")
    time_index = t0 + np.arange(n_samples) * dt_timedelta
    return time_index


def slow_comprehension(t0, n_samples, sample_rate):
    t0 = np.datetime64(t0)
    dt = 1.0 / sample_rate
    time_vector_seconds = dt * np.arange(n_samples)
    time_vector_nanoseconds = (np.round(1e9 * time_vector_seconds)).astype(int)
    time_index = np.array(
        [t0 + np.timedelta64(x, "ns") for x in time_vector_nanoseconds]
    )
    return time_index


TIME_AXIS_GENERATOR_FUNCTIONS = {}
TIME_AXIS_GENERATOR_FUNCTIONS["fast_arange"] = fast_arange
TIME_AXIS_GENERATOR_FUNCTIONS["slow_comprehension"] = slow_comprehension


def decide_time_axis_method(sample_rate):
    dt = 1.0 / sample_rate
    ns_per_sample = 1e9 * dt
    if np.floor(ns_per_sample) == np.ceil(ns_per_sample):
        method = "fast_arange"
    else:
        method = "slow_comprehension"
    return method


def make_time_axis(t0, n_samples, sample_rate):
    method = decide_time_axis_method(sample_rate)
    time_axis = TIME_AXIS_GENERATOR_FUNCTIONS[method](t0, n_samples, sample_rate)
    return time_axis


def test_generate_time_axis(t0, n_samples, sample_rate):
    """Two obvious ways to generate an axis of timestamps here. One method is slow and
    more precise, the other is fast but drops some nanoseconds due to integer
    roundoff error.

    To see this, consider the example of say 3Hz, we are 333333333ns between samples,
    which drops 1ns per second if we scale a nanoseconds=np.arange(N)
    The issue here is that the nanoseconds granularity forces a roundoff error


    Probably will use logic like:
    if there_are_integer_ns_per_sample:
        time_stamps = do_it_the_fast_way()
    else:
        time_stamps = do_it_the_slow_way()
    return time_stamps

    Parameters
    ----------
    t0
    n_samples
    sample_rate

    Returns
    -------

    """
    t0 = np.datetime64(t0)

    # SLOW
    tt = time.time()
    time_index_1 = slow_comprehension(t0, n_samples, sample_rate)
    processing_time_1 = tt - time.time()
    print(f"processing_time_1 = {processing_time_1}")
    

    # FAST
    tt = time.time()
    time_index_2 = fast_arange(t0, n_samples, sample_rate)
    processing_time_2 = tt - time.time()
    print(f"processing_time_2 {processing_time_2}")
    print(f"ratio of processing times {processing_time_1/processing_time_2}")
    
    if (np.abs(time_index_2 - time_index_1)).sum() == 0:
        pass
    else:
        print("Time axes are not equal")
    return time_index_1


# def make_time_axis_method1(t0, n_samples, sample_rate):
# pass


def do_some_tests():
    n_samples = 1000
    sample_rate = 50.0  # Hz
    t0 = pd.Timestamp(1977, 3, 2, 6, 1, 44)
    time_axis = test_generate_time_axis(t0, n_samples, sample_rate)
    print(f"{time_axis[0]} ...{time_axis[-1]}")
    sample_rate = 3.0  # Hz
    time_axis = test_generate_time_axis(t0, n_samples, sample_rate)


def main():
    do_some_tests()


if __name__ == "__main__":
    main()
