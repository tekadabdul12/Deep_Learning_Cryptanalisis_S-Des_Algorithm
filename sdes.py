#from tes_ascii import *

def apply_table(inp, table):
    """
    >>> apply_table("0123456789", list(range(10)))
    '9012345678'
    >>> apply_table("0123456789", list(range(9, -1, -1)))
    '8765432109'
    """
    res = ""
    for i in table:
        res += inp[i - 1]
    return res


def left_shift(data):
    """
    >>> left_shift("0123456789")
    '1234567890'
    """
    return data[1:] + data[0]


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
    return res


def apply_sbox(s, data):
    row = int("0b" + data[0] + data[-1], 2)
    col = int("0b" + data[1:3], 2)
    return bin(s[row][col])[2:]


def function(expansion, s0, s1, key, message):
    left = message[:4]
    right = message[4:]
    temp = apply_table(right, expansion)
    temp = XOR(temp, key)
    l = apply_sbox(s0, temp[:4])  # noqa: E741
    r = apply_sbox(s1, temp[4:])
    l = "0" * (2 - len(l)) + l  # noqa: E741
    r = "0" * (2 - len(r)) + r
    temp = apply_table(l + r, p4_table)
    temp = XOR(left, temp)
    return temp + right


if __name__ == "__main__":
    #listx = ascii_input()
    listx = ['00100000', '00100001', '00100010', '00100011', '00100100', '00100101', '00100110', '00100111',
             '00101000', '00101001', '00101010', '00101011', '00101100', '00101101', '00101110', '00101111',
             '00110000', '00110001', '00110010', '00110011', '00110100', '00110101', '00110110', '00110111',
             '00111000', '00111001', '00111010', '00111011', '00111100', '00111101', '00111110', '00111111',
             '01000000', '01000001', '01000010', '01000011', '01000100', '01000101', '01000110', '01000111',
             '01001000', '01001001', '01001010', '01001011', '01001100', '01001101', '01001110', '01001111',
             '01010000', '01010001', '01010010', '01010011', '01010100', '01010101', '01010110', '01010111',
             '01011000', '01011001', '01011010', '01011011', '01011100', '01011101', '01011110', '01011111',
             '01100000', '01100001', '01100010', '01100011', '01100100', '01100101', '01100110', '01100111',
             '01101000', '01101001', '01101010', '01101011', '01101100', '01101101', '01101110', '01101111',
             '01110000', '01110001', '01110010', '01110011', '01110100', '01110101', '01110110', '01110111',
             '01111000', '01111001', '01111010', '01111011', '01111100', '01111101', '01111110']

    key = input("Enter 10 bit key: ")

    for i in listx:
        message = i

        p8_table = [6, 3, 7, 4, 8, 5, 10, 9]
        p10_table = [3, 5, 2, 7, 4, 10, 1, 9, 8, 6]
        p4_table = [2, 4, 3, 1]
        IP = [2, 6, 3, 1, 4, 8, 5, 7]
        IP_inv = [4, 1, 3, 5, 7, 2, 8, 6]
        expansion = [4, 1, 2, 3, 2, 3, 4, 1]
        s0 = [[1, 0, 3, 2], [3, 2, 1, 0], [0, 2, 1, 3], [3, 1, 3, 2]]
        s1 = [[0, 1, 2, 3], [2, 0, 1, 3], [3, 0, 1, 0], [2, 1, 0, 3]]

        # key generation
        temp = apply_table(key, p10_table)
        left = temp[:5]
        right = temp[5:]
        left = left_shift(left)
        right = left_shift(right)
        key1 = apply_table(left + right, p8_table)
        left = left_shift(left)
        right = left_shift(right)
        left = left_shift(left)
        right = left_shift(right)
        key2 = apply_table(left + right, p8_table)

        # encryption
        temp = apply_table(message, IP)
        temp = function(expansion, s0, s1, key1, temp)
        temp = temp[4:] + temp[:4]
        temp = function(expansion, s0, s1, key2, temp)
        CT = apply_table(temp, IP_inv)
        print("Cipher text is:", CT)

        # decryption
        temp = apply_table(CT, IP)
        temp = function(expansion, s0, s1, key2, temp)
        temp = temp[4:] + temp[:4]
        temp = function(expansion, s0, s1, key1, temp)
        PT = apply_table(temp, IP_inv)
        print("Plain text after decypting is:", PT)