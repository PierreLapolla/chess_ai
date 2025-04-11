from utils.logger import log


def try_except(
    _func=None, *, finally_callable: callable = None, error_callable: callable = None
) -> callable:
    """
    Decorator to catch exceptions in a function.
    Optionally execute a callable in the finally block or when an error is caught.

    :param _func: The function to decorate.
    :param finally_callable: Optional callable to execute in finally.
    :param error_callable: Optional callable to execute when an error is caught.
    :return: The wrapper function.
    """

    def decorator_func(func: callable) -> callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                message = f"{func.__name__}: {e.__class__.__name__}: {e}"
                log.error(message)
                if error_callable:
                    error_callable(e)
            finally:
                if finally_callable:
                    finally_callable()
                    log.debug(f"Executed finally callable: {finally_callable.__name__}")

        return wrapper

    if _func is None:
        return decorator_func
    else:
        return decorator_func(_func)


if __name__ == "__main__":
    """
    Example usage of the try_except decorator with both error and finally callables.
    """

    def cleanup():
        print("Cleanup executed.")

    def handle_error(e):
        print(f"Error handled: {e}")

    @try_except(finally_callable=cleanup, error_callable=handle_error)
    def divide(a, b):
        return a / b

    divide(1, 0)
