def empty_copy(obj):
    '''
    Create a blank copy of an object without calling the __init__ method.
    # https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s12.html
    '''
    class Empty(obj.__class__):
        def __init__(self): 
            pass
    newcopy = Empty()
    newcopy.__class__ = obj.__class__
    return newcopy