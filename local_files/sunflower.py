# foods=list()
# while True:
#     food=input("what is your car name : ")
#     if food=="no":
#         print("no more your car")
#         break
#     foods.append(food)

# car=list()
# while(cars:=input("enetr your car name : "))!="no":
#     car.append(cars)

# foods=list()
# while (food:=input("enter your food name : ")) !="pizza":
#     foods.append(food)

# sachin = str(input("enetr your car name : "))
# for i in sachin.upper():
#     print(i.capitalize())
# print(sachin)

# print("*****")
# print("****")
# print("***")
# print("**")
# print("*")

# Define the initial number of stars
# n = 5

# # Use a for loop to iterate from n to 0
# for i in range(n, 0, -1):
#     # Print i stars in each line
#     print("#" * i)

# car=list()
# while (cars:=input("enter your car list :"))!="bugati":
#     car.append(cars)

# import functools

# @functools.lru_cache(maxsize=None)
# def fib(n):
#     if n < 2:
#         return n
#     return fib(n-1) + fib(n-2)

# print(fib(20))

# class sachin:
#     def __init__(self,):
#         pass

#     def masti(self):
#         s=list()
#         while True:
#             m=input("enter your name : ")
#             if m=="sachin":
#                 print("you are a awsome")
#                 break
#             else:
#                 print("cary on")
#         while True:
#           p=input("enetr your car name : ")
#           if p=="bugati":
#                 print("this is very fast")
#                 break
#         else:
#                 print("move on")
#         s.append(m)
#         s.append(p)
            

# cricket=sachin()
# cricket.masti()

# class bmw(sachin):
#     def __init__(self):
#         super().__init__()

# mustang=bmw()
# mustang.masti()

# from typing import Any


# class cricket:
#     def __init__(self) -> None:
#         pass
    
#     def mclaren(self):
#         my_list = list()
#         while True:
#             tennis=input("enter your game name : ")
#             if tennis=="kho-kho":
#                 print("this game is not valid🙃")
#                 break
#             else:
#                 print("cary on 👍")
#             my_list.append(tennis)
#         while True:
#             sports=input("enetr your sports name : ")
#             if sports=="polo":
#                 print("this is not valid game (●'◡'●)")
#                 break
#             else:
#                 print("cary on 👌")
#             my_list.append(sports)

# sun=cricket()
# sun.mclaren()
# sun.__call__()

# import requests
# url = "https://images.unsplash.com/photo-1694547278143-4c83c9d46b1e?w=900&auto=format&fit=\
# crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHx0b3BpYy1mZWVkfDZ8Ym84alFLVGFFMFl8fGVufDB8fHx8fA%3D%3D"
# response = requests.get(url)
# with open("unsplash.jpg", "wb") as f:
#     f.write(response.content)


# s={
#     "bugati":"nan",
#     "ferrari":"nan",
#     "meclaren":"nan",
#     "bmw":"nan"
# }
# m={
#     "bugati":input("eneter your car name : "),
#     "ferrari":input("enetr you name : "),
#     "meclaren":input("enetr your car name : "),
#     "bmw":input("enetr your car name : ")
# }
# sm=s|m
# print(sm)

# name=["bugati","lamborgini","porche","mclaren","koeingsega"]
# print(max(name,key=len))

# *_,="sachin123456"
# print(dict(),_)


# class sachin:
#     car_count=0
#     def __init__(self,car_name,car_price):
#         self.car_name=car_name
#         self.car_price=car_price
#         sachin.car_count+=1
    
#     def info(self):
#         print(f"the car no {self.car_count} is {self.car_name} and car price is {self.car_price}")

# masti=sachin("bugati","5m")
# masti.info()

# class cars:
#     car_name = input("enter your car name : ")
#     car_no = 0
#     car_price = input("enetr your car price : ")
#     def __init__(self):
#         self.car_country = input("enetr your car country name : ")
#         self.car_name
#         cars.car_no +=1
#         if self.car_name=="bugati":
#             print("this is insane❤️‍🔥")

#     def info(self):
#         print(f"the car no is {self.car_no} \nand the car name is {self.car_name
#         } and the car price is {self.car_price} and the  car come from {self.car_country}")
#         if self.car_country=="india":
#             print("this is my country❤️")

# sachin = cars("bugati")
# sachin.car_country 
# sachin.car_price
# sachin.info()

# masti=cars("nothing")
# masti.car_country
# masti.car_price
# masti.info()


class Vehicle:
    car_no = 0

    def __init__(self, car_name, car_power):
        self.car_name = car_name
        if car_name == "bugati":
            print("this is awesome🥰")
        self.car_power = car_power
        if car_power < str(1000):
    # do something

            print("this is beast👽")
        Vehicle.car_no += 1


    def show(self):
            print(f"the car no is {self.car_no} and the car is {self.car_name} \n the car power is {self.car_power}")
foot=Vehicle("bugati","1500hp")
foot.show()

cricket=Vehicle("mustang","500hp")
cricket.show()

import requests
import time
def sachin():
        time.sleep(3)
        url="https://th.bing.com/th/id/OIG.MHUCnvFcAR4_5FlQiaec?w=1024&h=1024&rs=1&pid=ImgDetMain"
        response=requests.get(url)
        open("aiimage.jpg","wb").write(response.content)
sachin()

import requests
import shutil

# The URL of the image to download
img_url = "https://th.bing.com/th/id/OIG.W6FvKkgFXaS0N4uHxH6H?w=1024&h=1024&rs=1&pid=ImgDetMain"

# The path where the image will be saved
path = "download1.png"

# Get the raw response from the URL
response = requests.get(img_url, stream=True)

# Open a file in write-binary mode
with open(path, "wb") as out_file:
    # Copy the raw response to the file
    shutil.copyfileobj(response.raw, out_file)

# Delete the response object to free up resources

