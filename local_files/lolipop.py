def gmean(a, b):
    mean= (a * b) / (a + b)
    print('The geometric mean is:', mean.__round__(2))
a = int(input('enter your number: '))
b = int(input('enter your number: '))
gmean1=(a*b)/(a+b)
print('The geometric mean is:', gmean1.__round__(2))

c=100
d=101
gmean(c,d)