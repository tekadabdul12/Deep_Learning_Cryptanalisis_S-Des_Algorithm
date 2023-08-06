
def ascii_input():
    text = input("enter a string to convert into ascii values:")
    ascii_values = []
    for character in text:
        ascii_values.append(ord(character))
    print(ascii_values)

    asbiner = []
    for i in ascii_values:
        print(i)

        asbiner.append("{0:08b}".format(int(i)))

        #print(asbiner)
    print(asbiner)
    return asbiner





