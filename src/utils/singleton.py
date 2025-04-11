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


if __name__ == "__main__":
    """
    Example usage of the singleton pattern.
    """

    @singleton
    class SingletonExempleClass:
        def __init__(self, value):
            self.value = value

    singleton_1 = SingletonExempleClass(0)
    singleton_2 = SingletonExempleClass(1)
    assert singleton_1 == singleton_2
    assert singleton_1.value == singleton_2.value

    class SingletonExempleClass(metaclass=SingletonMeta):
        def __init__(self, value):
            self.value = value

    singleton_1 = SingletonExempleClass(0)
    singleton_2 = SingletonExempleClass(1)
    assert singleton_1 == singleton_2
    assert singleton_1.value == singleton_2.value
