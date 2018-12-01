from z3 import *

def main():
    l = Int('the rank of Lisa')
    j = Int('the rank of Jim')
    b = Int('the rank of Bob')
    m = Int('the rank of Mary')

    s = Solver()
    s.add(0 < l, l < 5, 0 < j, j < 5, 0 < b, b < 5, 0 < m, m < 5)
    s.add(Not(l == j), Not(l == b), Not(l == m), Not(j == b), Not(j == m), Not(b == m))

    s.add(Or(l - b > 1, b - l > 1))
    s.add(Or(j - l == -1, j - m == -1))
    s.add(b - j == -1)
    s.add(Or(l == 1, m == 1))

    print('Check =', s.check())
    print('The solution is:')
    print(s.model())

if __name__ == '__main__':
    main()