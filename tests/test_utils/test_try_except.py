from utils.try_except import try_except


def test_try_except_decorator(caplog):
    flag = {"cleanup_executed": False}

    def cleanup():
        flag["cleanup_executed"] = True

    @try_except(finally_callable=cleanup)
    def divide(a, b):
        return a / b

    flag["cleanup_executed"] = False
    result = divide(4, 2)
    assert result == 2, "Expected 4 / 2 to equal 2"
    assert flag["cleanup_executed"] is True, "Expected finally_callable to be executed"

    flag["cleanup_executed"] = False
    result = divide(1, 0)
    assert result is None, "Expected 1 / 0 to return None due to exception"
    assert flag["cleanup_executed"] is True, (
        "Expected finally_callable to be executed even on exception"
    )


def test_error_callable():
    flag = {"error_called": False, "cleanup_executed": False}
    exception_message = {"msg": ""}

    def cleanup():
        flag["cleanup_executed"] = True

    def handle_error(e):
        flag["error_called"] = True
        exception_message["msg"] = str(e)

    @try_except(finally_callable=cleanup, error_callable=handle_error)
    def divide(a, b):
        return a / b

    flag["error_called"] = False
    flag["cleanup_executed"] = False
    result = divide(1, 0)
    assert result is None, "Expected divide to return None on exception"
    assert flag["error_called"] is True, (
        "Expected error_callable to be executed on exception"
    )
    assert flag["cleanup_executed"] is True, (
        "Expected finally_callable to be executed on exception"
    )
    assert "division by zero" in exception_message["msg"], (
        "Expected division by zero error message"
    )
