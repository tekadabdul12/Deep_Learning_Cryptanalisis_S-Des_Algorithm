def XOR(a, b):
    """
    >>> XOR("01010101", "00001111")
    '01011010'
    """
    res = ""
    for i in range(len(a)):
        if a[i] == b[i]:
            res += "0"
        else:
            res += "1"
        print(i,a,b, res)
    return res


print(XOR("01010101", "00001111"))
