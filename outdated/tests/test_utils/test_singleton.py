from utils.singleton import singleton, SingletonMeta


def test_singleton_decorator():
    @singleton
    class TestSingletonDecorator:
        def __init__(self):
            pass

    instance1 = TestSingletonDecorator()
    instance2 = TestSingletonDecorator()
    assert instance1 is instance2, (
        "The singleton decorator did not return the same instance"
    )


def test_singleton_meta():
    class TestSingletonMeta(metaclass=SingletonMeta):
        def __init__(self):
            pass

    instance1 = TestSingletonMeta()
    instance2 = TestSingletonMeta()
    assert instance1 is instance2, (
        "The singleton metaclass did not return the same instance"
    )
