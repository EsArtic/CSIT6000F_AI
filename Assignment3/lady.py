from z3 import *

def main():
    sign1 = Bool('ROOM I: A TIGER IS IN THIS ROOM.')
    sign2 = Bool('ROOM II: A LADY IS IN THIS ROOM.')
    sign3 = Bool('ROOM III: A TIGER IS IN ROOM II.')

    l1 = Bool('lady is in room 1.')
    l2 = Bool('lady is in room 2.')
    l3 = Bool('lady is in room 3.')

    print('Denote \'%s\' by sign1.' % sign1)
    print('Denote \'%s\' by sign2.' % sign2)
    print('Denote \'%s\' by sign3.' % sign3)
    print('Denote \'%s\' by l1.' % l1)
    print('Denote \'%s\' by l2.' % l2)
    print('Denote \'%s\' by l3.' % l3)
    print()

    s = Solver()
    '''
        One room contained a lady and the other two contained tigers.
        E.g. l1 implies not(l2) --  not(l1) or not(l2) ...
    '''
    s.add(Or(Not(l1), Not(l2)), Or(Not(l1), Not(l3)), Or(Not(l2), Not(l3)))

    '''
        There must have one room contained a lady.
    '''
    s.add(Or(l1, l2, l3))

    '''
        At most one of the three signs was true.
        E.g. sign1 implies not(sign2) -- not(sign1) or not(sign2) ...
    '''
    s.add(Or(Not(sign1), Not(sign2)), Or(Not(sign1), Not(sign3)), Or(Not(sign2), Not(sign3)))

    '''
        Other constraints: The relation between signs and l1, l2, l3.
    '''
    s.add(Or(Not(sign1), Not(l1)), Or(sign1, l1))
    s.add(Or(Not(sign2), l2), Or(Not(l2), sign2))
    s.add(Or(Not(sign3), Not(l2)), Or(sign3, l2))

    try_list = [l1, l2, l3]
    answer = None
    for i in range(3):
        print('Check whether Not(l%d) is consistent with the KB.' % (i + 1))
        s.push()
        s.add(Not(try_list[i]))
        check = s.check()
        if check == sat:
            print('    return sat, l%d isn\'t a correct answer.' % (i + 1))
        else:
            answer = try_list[i]
            print('    return unsat, l%d is a correct answer.' % (i + 1))
        print()
        s.pop()

    if answer != None:
        print('The answer is: %s' % answer)
        print('One of the solution is:')
        s.add(answer)
        print(s.model())

if __name__ == '__main__':
    main()