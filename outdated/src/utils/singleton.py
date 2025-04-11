from functools import wraps


def singleton(cls):
    """
    Decorator to implement the singleton pattern.

    :param cls: The class to be decorated.
    :type cls: type
    :return: The singleton instance of the decorated class.
    :rtype: object
    """
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        """
        Retrieve the singleton instance of the decorated class.

        :param args: Positional arguments for cls initialization.
        :param kwargs: Keyword arguments for cls initialization.
        :return: Singleton instance of cls.
        """
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class SingletonMeta(type):
    """
    Metaclass implementing the singleton pattern.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Control the instance creation for classes using this metaclass.

        :param args: Positional arguments for cls initialization.
        :param kwargs: Keyword arguments for cls initialization.
        :return: Singleton instance of cls.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
