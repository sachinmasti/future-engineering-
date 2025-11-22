# Calculate the sum of all digits from a string.
'''lst = 'a1b2c3'
sum_all = 0
# Loop through each character in the string.
for i in lst:
    # Check if the character is a digit.
    if i.isdigit():
        # If it's a digit, convert it to an integer and add to `sum_all`.
        sum_all+=int(i)
# Print the final sum.
print(sum_all) 

# Replace a character in a string with another character.
string = 'banana'
# Replace 'a' with 'x' and print the new string.
print(string.replace('a','x'))

# Remove duplicate items from a list.
s =[1,2,3,4,5,6,44,2323,34,34,54,56,5,43,233,23,34,34,5454,23,2323,23]
# Remove duplicates using `dict.fromkeys()` and then convert back to a list.
print(list(dict.fromkeys(s)))

# Import the math module as 'm'.
import math as m
n1 =4
n2= 6

# Calculate the Least Common Multiple (LCM) of two numbers.
# LCM(a, b) = (|a * b|) / GCD(a, b)
lscm = (n1 * n2) / m.gcd(n1,n2)
print(int(lscm))

#
# from colorama import Fore,Style,init
# init(autoreset=True)
# num = 10
# all_num = []
# sum_num = 0
# for i in range(1,num):
#     if i % 2==0:
#         sum_all +=i
#         all_num.append(i)
# print(all_num)
#
# if sum_all % num ==0:
#     print(f'{num} {Fore.CYAN}is a strong number')
# else:
#     print(f'{num}{Fore.MAGENTA} is not a strong number')


num1 = 12345678910
reversed_num = 0
 
while num1 > 0:
    # Get the last digit
    digit = num1 % 10
    # Append the digit to the reversed number
    reversed_num = (reversed_num * 10) + digit
    # Remove the last digit from the original number
    num1 //= 10
print(reversed_num)


# def is_balanced(string):
#     # opeing = ['(','[','{']
#     colsing = [')',']','}']
#     for i in string:
#         for j in colsing:
#             if i =='(' and j ==')':
#                 return i + j
#             elif i =='[' and j ==']':
#                 return i+j
#             elif i =='{' and j =='}':
#                 return i + j
            
# sm = '{'
# print(is_balanced(sm))

def is_balsnced(string):
    stack = []
    opening = "({["
    closing = ")}]"
    pairs = {')': '(', '}': '{', ']': '['}

    for  i in string:
        if i in opening:
            stack.append(i)
        
        elif i in closing:
            if not stack or stack[-1] !=pairs[i]:
                return False
            stack.pop()
    return len(stack)==0

test_cases = [
    "{[()]}",
    "{[(])}",
    "((()))",
    "({[}])",
    "",
    "{[}",
]

print("Bracket Matching Check:")
print("="*50)
for test in test_cases:
    result = "✓ Balanced" if is_balsnced(test) else "✗ Not Balanced"
    print(f"{test:15} → {result}")

print("\n" + "="*50 + "\n")

l : list = [1,2,3,5,7,8,9,4,2]
target = 3
for i in range(1,len(l)):
    if l[i] + l[i-1] == target:
        print(l[i-1] , l[i])
'''
print('hi my name is sachin😈')

# row = 6
# for i in range(row):
#     for j in range(row):
#         if i ==j or j ==row-1-i:
#             print(' ',end='')
#         else:
#             print('*')

import time
import math as m
start = time.time()
num = 100003
if num <=1:
    print('this is not a prime number')

elif num ==2:
    print('this is a prime number')

elif num % 2==0:
    print('this is a prime nubmer')
else:   
    for i in range(3,int(num **0.5 )+1,2):
        if num % i == 0:
            print('this is not a prime number')
            break
    else:
        print('this is a prime number')



end = time.time()



print("Time taken:", end - start, "seconds")