'''print('sachin masti')

for i in range(1,11):
    print(i,end=' ')

word = input('enter your name: ')
my_dict = {}
for char in word:
    if char in my_dict:
        my_dict[char]+=1
    else:
        my_dict[char]=1
for key,values in my_dict.items():
    print(f'this word is {key} : {values} times')

for i in range(1,100,2):
    print(i)


import math as m
a = 193
b = 386

print(m.factorial(a))
print(m.sqrt(b))
print(m.fabs(b))

import random
random_number=random.randint(1,100)

while True:
    number=int(input('enter your number: '))
    if number < 1 or number >100:
        print('this number is not valid plz enter\n a number number betwen 1 to 100')
        continue

    if number == random_number:
        print('you guesed right⭐')
        break
    
    elif number < random_number:
        print('number is to low! try again')
    
    else:
        print('number is high try again')
        

car_list=['bugati','ferrari','bbw','porche']
new_list=[]
for c in car_list:
    new_list.append(c[0])
    print(c)

x=int(input('enter your x: '))
y=int(input('enter your y: '))
z=int(input('enter your z: '))
n=int(input('enter your n: '))
result = [
    [i,j,k]
          for i in range(x+1)
          for j in range(y+1)
          for k in range(z+1)
          if i+j+k !=n]
print(result)'''

name='sachin masti!!!!'
print(name.rstrip('!'))
print(name.endswith('!'))
print(name.center(50))
print(name.find('ma'))
print(name.count('m'))
print(name.capitalize())
print(name.title())
print(name.upper())
print(name.index('c'))
print(name.find('t'))

import time
msg=time.strftime('%H:%M:%S')
msg=time.strftime('%H')
print(msg)
msg=time.strftime('%M')
print(msg)
msg=time.strftime('%S')
print(msg)

import time
time_input=int(input('enter your time: '))
now_time=time.localtime().tm_hour
if time_input >12:
    print('good morning')
elif time_input>=12 and time_input <=3:
    print('good afternoon')
elif time_input >=3 and time_input <=6:
    print('good noon')
elif time_input >=6 :
    print('good night')
else :
    print('time to do it')


import time

now_time = time.localtime().tm_hour

if 5 <= now_time < 12:
    print('Good morning')
elif 12 <= now_time < 17:
    print('Good afternoon')
elif 17 <= now_time < 21:
    print('Good evening')
else:
    print('Good night')

def calulate():
    weight = int(input('enter your weight: '))
    hight = int(input('enter your hight: '))
    result1= weight/(hight**2)
    if result1 >=80: print('over weight')
    if result1>=55 and result1<=65: print('you are a fit')
    if result1 <=45: print('you are a under weight')
final=calulate()

name=input('enter your name: ')
match name:
    case  'sachin':
        print('you are a ⭐')
    case 'shirish':
        print('you are a awsome')
    case 'girish':
        print('hi you are a creazy')
    case _:
        print('name not found')

name='harry'
print(name[-4:-2])

import this

num=[1,3,44,33,55,66,44,654,456,43]
result = min(num) 
resul1=max(num)
print(result , resul1)

name_ist='1600 bugati is the worlds fastest car in the car leagcy '
s_spit= name_ist.split()
sm=int(name_ist[0])
print(sm)
length= [len(words) for words in s_spit]
small = min(length)

print(f'this is a smallest world in {small}')

# Function to calculate the factorial of a number using recursion
def factorial(num):
    if num == 1 or num == 0:  # Base case: factorial of 0 or 1 is 1
        return 1
    else:
        return num * factorial(num - 1)  # Recursive call

# Input from the user to calculate factorial
num = int(input('Enter your number: '))
boom = factorial(num)  # Call the factorial function
print(boom)  # Print the result

# Function to calculate the nth Fibonacci number using recursion
def fibonacchi(n):
    if n == 0 or n == 1:  # Base cases: Fibonacci of 0 or 1 is 1
        return 1
    else:
        return fibonacchi(n - 1) + fibonacchi(n - 2)  # Recursive call

# Input from the user to calculate Fibonacci number
n = int(input('Enter your number: '))
xyz = fibonacchi(n)  # Call the Fibonacci function
print(xyz)  # Print the result


name=['bugati','ferrari','bmw','porche','mustang']
letter = ['s','a','z']
count_letter = name.count(letter)
print(count_letter)



def sschin(s):
    s.lower()
    for i in ('abcdefghijklmnopqrstuvwxyz'):
        return False
    return True
s='the quick box jumps over the lazy dog'
ms=sschin(s)
print(ms)

def re(array):
    odd = sorted((x for x in array if x%2!=0), reverse=True)
    return [x if x%2==0 else odd.pop() for x in odd]

l=[43,23,45,32,23,12,4,4,12,11,21,12,2,1,3,4,65]
print(re(l))
print('sachin')


from colorama import Fore , Style
print(Fore.LIGHTYELLOW_EX + 'Hare Krisha 🌻')


with open(r"D:\zdenek-machacek-UxHol6SwLyM-unsplash.jpg", "rb") as f:
    content = f.read()



lst = [1, 2, 3, 4, 5, 6, 6, 7, 5, 4, 3, 3, 4, 5, 6]
result = list(map(lambda x: x * 5, lst))
print(result)



lst = [1,2,3,4,5,6,7,8,8,9]
# result = list(map(lambda x:x*x*x,lst))
# print(result)

# new = list(filter(lambda x : True if x>5 else False,lst))
# print(new)

from functools import reduce
new3 = reduce(lambda x,y:x+y,lst)
print(new3)


ns=map(lambda x:x*2,lst)
print(ns)



lst = [1,2,3,4,5,6,7,8,8,9]
lst2 = [34,5,3,5,6,7,7,5,3,2,567,8,6,4,3,2]
lst3 = [12,3,4212,3,46,4,67,8,7,3,2,77,6,5]
new_line1 = map(lambda x:x**2,filter(lambda x,y,z:x>5,lst,lst2,lst3))
print(list(new_line1))

