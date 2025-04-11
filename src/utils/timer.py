from time import perf_counter

from utils.logger import log


def timer(func: callable) -> callable:
    """
    Decorator to measure the execution time of a function.
    :param func: function to measure the execution time of
    :return: wrapper function that measures the execution time of the function
    """

    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        log.info(f"Execution time of {func.__name__}: {end - start} seconds")
        return result

    return wrapper


if __name__ == "__main__":
    """
    Example usage of the timer decorator.
    """

    @timer
    def example_function():
        for _ in range(1000000):
            pass

    example_function()
