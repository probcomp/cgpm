class A(object):
    def __init__(self):
        pass

    @staticmethod
    def farabans(x):
        print 'Farabans in A'

class B(A):
    def __init__(self):
        pass

    @staticmethod
    def farabans(x):
        print 'Farabans in B'

def hans(a, c, b):
    print (a,c,b)
