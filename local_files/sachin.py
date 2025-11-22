# for i in range (10):
#     print(i+1)

# i=int(input("enetr your number :"))
# match i:
#     case 1:
#         print("i is zero")
#     case _ if i==10:
#         print("i is euql to 10")
#     case _ if i>100:
#         print("this is big number.")
#     case _:
#         print(i)
#         print("end")

# while True:
#     i=int(input("enetr your number : "))
#     if (i < 10):
#         break
#     print(i)

# for i in[1,2,3,4,5,6,7,8,9,0]:
#     if (i%2!=0):
#         continue
#     print(i)

# sachin=10
# while(sachin > 0):
#     print(sachin)
#     sachin=sachin-1

# tup=(100,200,300,400)
# print(type(tup),tup)
# print(tup[2])
# print(tup[-1])
# tup2=tup[1:3]
# print(tup2)
# if 200 in tup:
#     print("200 is present in tup.")

# countries = ("Spain", "Italy", "India", "England", "Germany")
# temp = list(countries)
# temp.append("Russia")       #add item 
# temp.pop(3)                 #remove item
# temp[2] = "Finland"         #change item
# countries = tuple(temp)
# print(countries)

# countries=("india","japan","china","england","taiwan","vaitnam")
# temp = list(countries)
# temp.append("brazil")
# temp.pop(3)
# temp[5]="south africa"
# countries=tuple(temp)
# print(countries,"there are 'brics'countries")

# import time
# t=time.strftime('%H:%M:%S')
# hour=int(time.strftime("%H"))

# if (hour >= 0 and hour<12):
#     print("good morning friend.")
# elif (hour>=12 and hour<17 ):
#     print("good after noon friend.")
# if (hour>17 and hour<0):
#     print("good night friend")

# prices=[100,200,300]
# total = 0
# for items in prices:
#     total += prices
# print(f"total prices is :{total}") 

# num=[5,4,3,2,1]
# for count in num:
#     print('$'*count)

# number=[200,300,400,800,1000]
# max=number[0]
# for num in number:
#     if (num > max):
#         max=num
# print(num)

# def calculate(a,b):
#     c=(a-b)
#     print(c)

# def isgreterthen(a,b):
#     if (a > b):
#         print("a is greter then b")
#     else:
#         print("both are same number")

# x=100
# y=300
# calculate(x,y)
# isgreterthen(x,y)

# s=2000
# m=1000
# calculate(s,m)
# isgreterthen(s,m)


# d=100
# c=200
# calculate(d,c)
# isgreterthen(d,c)

# def factorial(num): 
#     if (num == 1 or num == 0):
#         return 1
#     else:
#         return (num * factorial(num - 1)) 
  
# # Driver Code 
# num = 7; 
# print("Number: ",num)
# print("Factorial: ",factorial(num))

# def calculate(a,b):
#     '''this is the doc str'''
#     num=(a*b)
#     print(num)

# x=10
# y=10
# calculate(x,y)
# print(calculate.__doc__)

# s = [1,2,3,4,5,1,4,5]
# sachin=[]
# for s in s:
#     if s not in sachin:
#         sachin.append(s)
# print(sachin)

phone=input("enetr your phone number : ")
sachin={
    '1':"one",
    '2':"two",
    '3':"three",
    '4':"four",
    '5':"five",
    '6':"six",
    '7':"seven",
    '8':"eait",
    '9':"nine",
    '10':"ten"
}
output=""
for ch in phone:
    output+=sachin.get(ch,"")+ " "
print(output)

# sports={"cricket","football","basketball"}
# sports1={"tenis","baseball","hockey","football"}
# sports.discard("basketball")
# if "cricket" in sports:
    # print("cricket is present")
# else:
    # print("this is noy here")
# print(sports)

# cities = {"Tokyo", "Madrid", "Berlin", "Delhi"}
# cities2 = {"Seoul", "Kabul"}
# print(cities.issuperset(cities2))
# cities3 = {"Seoul", "Madrid","Kabul"}
# print(cities.issuperset(cities3))

# info = {'name':'Karan', 'age':19, 'eligible':True}
# print(info['name'])

# for key ,value in info.items():
    # print(f"this is dict {key} :\n and this is work {value}")

# print("bugati",1000,1200,sep="-",end="-1300")
# print("ferrari")

# car="bugati"
# car1="ferrari"
# car2="mclaren"
# cars='''cars is the most beutifull 
# thing by man made
# "i love cars'''
# print("my fav car is ",car)
# print(car[0])
# print(car[1])
# print(car[2])
# print(car[3])
# print(car[4])
# print(car[5])
# for characters in cars:
#     print(characters)

# print("bugati is my fav car\nand is very \'beutifull\' car in the \'world\'")
# print("cars")

# while True:
#     num=int(input("enetr your number : "))
#     print(num)
#     if (num > 10):
#         print("done")
#         break


# cars={"bugati" : 1000,"ferrari" : 800 ,"mustang" : 750}
# del cars
# print(cars)
# cars.update({"mustang" : 800})
# cars.update({"bugati" : 1500})
# cars.items

# for x in range(5):
#     print ("iteration no {} in for loop".format(x+1))
# else:
#     print ("else block in loop")
# print ("Out of loop")

# for x in range(5):
#     print("itration no {}in for loop".format(x+1))
# else:
#     print("else block in loop")
# print("out of loop")

cars={"bugati" : 1500, "ferrari" :800, "mustng" : 500,"mclaren" :1000, "porche" :650, "toyota" :700, "bmw" :700, "honda" :100}
cars.update({"dodge" :870})
cars.pop('toyota')
cars.popitem()
del cars['ferrari']
# cars.clear()
print(cars)