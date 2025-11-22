# class poco:
#     count=0
#     def __init__(self,name,age):
#         self.name=name
#         if "sachin" in self.name:
#             print("hi budy👽")
#         self.age=age
#         poco.count+=1

#     def info(self):
#         print(f"the person no is {self.count} \nthe name is {self.name} and\n the age is {self.age}")
# sachin=poco("sachin","22")
# sachin.info()

# masti=poco("rohit","37")
# masti.info()

# # Define the class
# class poco:
#     count=0
#     def __init__(self,name,age):
#         self.name=name
#         if "sachin" in self.name:
#             print("hi budy👽")
#         self.age=age
#         poco.count+=1

#     def info(self):
#         print(f"the person no is {self.count} \nthe name is {self.name} and\n the age is {self.age}")

# # Initialize a variable to control the loop
# i = 0

# # Start the loop
# while i < 10:
#     # Create two instances of the class
#     sachin=poco("sachin","22")
#     sachin.info()

#     masti=poco("rohit","37")
#     masti.info()

#     # Increment the variable
#     i += 1


# class cars:
#     count = 0

#     def __init__(self):
#         pass

#     def info(self):
#         my_list = []
#         while True:
#             car_name = input("Enter your car name: ")
#             if "bugati" in car_name.lower():
#                 print("This is an insane car 🥶")
#                 break
#             else:
#                 print("Carry on 👉")
#             my_list.append(car_name)
#         while True:
#             car_power = input("Enter your car horse power: ")
#             try:
#                 car_power = int(car_power)
#                 if car_power >= 1500:
#                     print("This is insane 🥵")
#                     break
#                 else:
#                     print("Move on 👉")
#                 my_list.append(car_power)
#             except ValueError:
#                 print("Invalid input. Please enter a number.")
#         print(my_list)
#         return my_list

# sm = cars()
# sm.info()


# class marks:
#     count=0

#     def __init__(self) -> None:
#         pass
#     def student(self):
#        my_dict=[]
#        while True:
#            marks.count+=1
#            student_name=input("enetr student name : ")
#            print(self.count)
#            if "sachin" in student_name.lower():
#                print("you are good👌")
#                break
#            else:
#                print("carry on👉")
#            my_dict.append(student_name)

#        while True:
#            marks.count+=1
#            print(self.count)
#            mark=input("enter your marks : ")
#            try:
#                mark=int(mark)
#                if mark>=80:
#                    print("this is awsome🔥")
#                    break
#                else:
#                    print("cary on👉")
#                my_dict.append(marks)
#            except ValueError:
#                print("this is not valid ")
               
#        print(my_dict.__dict__)
#        return my_dict

# poco=marks()
# poco.student()          



# class marks:
#     count = 0

#     def __init__(self):
#         pass

#     def student(self):
#         my_dict = []
#         while True:
#             self.count += 1
#             student_name = input("Enter student name: ")
#             print(self.count)
#             if "sachin" in student_name.lower():
#                 print("You are good👌")
#                 break
#             else:
#                 print("Carry on👉")
#             my_dict.append(student_name)
#         while True:
#             print(self.count)
#             score = input("Enter your marks: ")
#             try:
#                 score = int(score)
#                 if score >= 80:
#                     print("This is awesome🔥")
#                     break
#                 else:
#                     print("Carry on👉")
#                 my_dict.append(score)
#             except ValueError:
#                 print("This is not valid")
#         print(my_dict)

# poco = marks()
# poco.student()


# import PyPDF2 
# sachin=open("D:\\The_Practices_of_Programming.pdf","rb")
# sm=PyPDF2.PdfReader(sachin)
# for page in sm.pages:

#  print(page.extract_text())


# import PyPDF2


temp=34
massege="to hot " if temp>30 else"enjoy day"
print(massege)

# class cars:
#     car_no=0
#     def __init__(self) -> None:
#         pass
#     def car(self):
#         car_list=list()
#         while True:
#                 car_no=+1
#                 car_name=input("enter your car car name : ")
                
#     def car_power(self):
#          while True:
#               car_power=input("enter your car power : ")

# sachin=cars()
# sachin.car
# sachin.car_power

# rows=int(input("enter your rows : "))
# column=int(input("enter your column : "))
# symbols=(input("enter your symbols : "))

# for i in range(rows):
#     for y in range(column):
#         print(symbols, end="")
#     print()

# def addition(x):
#     if x<=50: return "c"
#     if x>=60: return "B"
#     if x >=70: return "B+"

# print(addition (66))

# from typing import Any
# class sachin:
#     count=0
#     def __init__(self) -> None:
#         pass

#     def info(self):
#         sm=[]
#         while True:
#           sachin.count+=1
#           name=input("enter your name : ")
#           if "sachin" in name.lower():
#               print("you are awsome 🥵")
#               break
#           else:
#               print("carry on 🔥")
#               sm.append(name)
#         print(sm)
#         while True:
#             marks=input("enter your marks : ")
#             marks=int(marks)
#             if marks>= 90:
#                 print ("your are awsome 🙃")
#                 break
#             else:
#                 print("carry on 👌")
            
#             sm.append(marks)
#         print(sm)

# sm=sachin()
# sm.info()        

# car_list=["bugati","ferrari","bmw","mclaren"]

# for i,thing in enumerate(car_list):
#     print(i+1,thing.title())

# car=["bugati","ferrari","mclaren","mercedes"]
# for i,things in enumerate(car):
#     print(i+1,things.title())

# def do_this():
#     print("do this")

# def do_that():
#     print("do that")

# while True:
#     choice=input("enter your chice : ")
#     match choice:
#         case "this":
#             do_this()
#         case "that":
#             do_that()
#         case _:
#             print("invalid input")
#             break

name:str="sachin masti"
print(f"{name = }")

# name=input("enter your name : ")
# new_name=name or "N/A"
# print(new_name)

# nums=range(1,100)

# def is_prime(num):
#     for i in range(2,num):
#         if (num%i) == 2:
#             return False
#         return True
    
# primes=list(filter(is_prime,nums))
# print(primes)

a=[1,2,3,4,5,6,7,8,9]
b=[1,2,3,4,5,6,7,8,9]
def list_sorted(arry1,arry2):
    return sorted(set(arry1+arry2))
c=list_sorted(a,b)
print(c)