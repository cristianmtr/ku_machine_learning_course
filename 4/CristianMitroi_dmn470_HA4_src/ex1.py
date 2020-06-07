def ex1():
    print("ex 1...")
    p = 1
    showups = 9600
    total = 10100
    for i in range(100):
        p = p * (showups-i)/(total-i)
    print('ex1: ', p)


if __name__ == "__main__":
    ex1()
