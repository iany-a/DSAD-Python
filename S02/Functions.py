def swap(a,b):
    print(a, 'before the swap',id(a))
    print(b, 'before the swap', id(b))
    aux = a
    a = b
    b = aux
    print(a, 'after the swap',id(a))
    print(b, 'after the swap', id(b))

    return None #need to return something

def swap_2(something):
    something[0], something[1] = something[1], something[0]
    return None