__author__ = 'asuiu'


class ValidateError(ValueError):
    def __init__(self, args):
        ValueError.__init__(self, args)


def validate(expr, msg="Invalid argument", exc: Exception = ValidateError):
    """
    If the expression val does not evaluate to True, then raise a ValidationError with msg
    """
    if not expr:
        raise exc(msg)
