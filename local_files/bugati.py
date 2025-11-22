# try:
#     num = int(input("Enter an integer: "))
#     a = [6, 3]
#     print(a[num])
# except ValueError:
#     print("Number entered is not an integer.")
    
# except IndexError:
#   print("Index Error"


# s=input("enter your number : ")
# print(f"multification table {s} is: ")
# try:
#     for i in range(1,11):
#         print(f"{int(s)} X {i} = {int(s) * i}")
# except :
#     print("invalid number ")

# finally:
#     print("its me 🤣")



# def func1():
#     try:
#      l=[100,200,300,400,]
#      i=int(input("enter your index  : "))
#      print(l[i])
#      return 1
#     except:
#        print("some error ocurerd ")
#        return 0
#     finally:
#        print("Done 🙃")
# x=func1()
# print(x)


# sachin=int(input("enetr your index : "))
# if (sachin<100 or sachin> 200):
#     raise("invalid index")


# a = 330000
# b = 3303
# print("A") if a > b else print("=") if a == b else print("B")

# c = 9 if a>b else 0
# print(c)

# srting=input("enetr your string : ")
# count=0
# for char in srting:
#     if char in "sachin":
#         count+=1
#     print("the number of",count)

# string = input("Enter a string: ")

# # Initialize a count variable to 0
# count = 0

# # Loop through each character in the string
# for char in string:
#     # Check if the character is a vowel
#     if char in "aeiouAEIOU":
#         # If it is, increment the count
#         count += 1

# # Print the count of vowels
# print("The number of vowels in the string is:", count)



# for i in range(2,9,2):
#     print(i)

# adj = ["red", "big", "tasty"]
# fruits = ["apple", "banana", "cherry"]
# for x in adj:
#   for y in fruits:
#     print(x, y)





# def factorial (n):
#     if n==1:
#         return 1
#     else:
#         return(n * factorial(n-1))
# num=6
# result=factorial(num)
# print("the factorial of",num, "is",result)

# def sachin():
#     try:
#      l=[100,200,300,400,500]
#      i=int(input("enter your index : "))
#      print(l[i])
#     except:
#        print("invalid index")
#     finally:
#        print("🎶")
# s=sachin()
# print(s)




# def sachin():
#     try:
#         l=["bugati","ferrari","mclaren","dodge"]
#         s=int(input("enetr your index : "))
#         print(l[s])
#         return 1
#     except:
#         print("invalid index ")
#         return 0
#     finally:
#         print("nice try 👌")
# d=sachin()
# print(d)


# def sachin(a,b):
#     print(a+b)
#     if a>b:
#         print("a is bigger then b")
#         return 1
#     else:
#         print("b is greter then a")
#         return 0
# x=1000
# z=200
# sachin(x,z)

# cars=["bugati","ferrari","mclaren"]
# for index,car in enumerate(cars,start=1):
#     print(car)
#     if (index==1):
#         print("lets go fast")

# import pandas as pd
# print(dir(pd))
# print(type(pd.test))