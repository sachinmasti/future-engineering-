'''for x in range(1,100):
    if x == 25:
            print('this is a our mid point🏆')
    if x == 50:
        print('our goal is a rach🦄')
        break
    else:
        print(x)'''

# name=(input('enter your cae name :👉 '))
# print(f"my car name {name}")
# print(type(name))

cars =10
print(cars % 4)

if (0.1 + 0.2 == 0.3):
     print('true')
else:
     print('false')
'''import math
user_input = float(input('enter circle radius: '))
new_input=2 * math.pi * user_input
print(f'the circumference of circle is {round(new_input)}')
'''
# print(help(str))

game_name='cricket'
print('this is my fav game💕' if game_name is 'cricket' else ' this is a not my fav game🙃')

new_game='this is my fav game and very beutifull' if game_name is 'cricket' else 'this is a good game '
print(new_game)    


name_of_game='cricket'
print('cricket is a my fav game❤️' if name_of_game is 'cricket' else 'this is not a my fav game🙃')

new_name ='this is a my most fav game in the world😍' if name_of_game is 'cricket' else ' this is a not a my game🌊'
print(new_name)

# name=input('enter your name')
# while name =="":
#      print('you not enter your name')
#      name=input('Enter your name:👉  ')
# print(f'hello {name}')


# food=input('enter your food or v to quite: ')
# while not food =='v':
#      print('you like a',food)
#      food=input("enter your fav food: ")
# print('bye')


# rows=int(input("enter your rows no:👉  "))
# coloum=int(input("enter your coloum no:👉  "))
# symbols=input("enter your symbols👉  ")
# for i in range(rows):
#      for j in range(coloum):
#           print(symbols,end="")
#      print()


# row=int(input("enter your rows:🪸 "))
# coloum=int(input("enter your coloum:🦑 "))
# symbol=input("enter your symbol:🏆 ")
# for i in reversed(range(row)):
#      for j in reversed(range(coloum)):
#           print(symbol,end="")
#      print()

if True+2==3:
     print('true')
else:
     print('false')
     


# import time
# name=input('enter your game name: ')
# play='this is a very beutifull game' if name == 'cricket' else 'this is a goodd game'
# time.sleep(5)
# print(play)







'''


import random

low_num=1
high_num=100
gusse=0
number=random.randint(low_num,high_num)
is_running=True
print('⭐------------welcome the Game------------⭐')
print(f'enter the number betwen {low_num} and {high_num}')

while is_running:
     your_num=input('enter your number ')
     if your_num.isdigit():
          your_num=int(your_num)
          gusse+=1

          if your_num <  low_num or your_num > high_num:
               print('number out of the range')
               print(f'enter number betwen { low_num} to {high_num}')
          elif gusse < number:
               print('number is to low')
          elif gusse > number:
               print('number is to high')

          else:
                  print(f"correct! the answer was {number}")
                  print(f"you took {gusse} guesses")
                  is_running = False

     else:
            print("invalid guessed")
            print(f"enter your number between {low_num} to {high_num}")

student={'sachin':'A',
         'shirish':'A+',
         'shramik':'A++',
         'girish':'A+',
         'prakash':'A+'}

your_output=input('enter a student name: ')
if your_output in student:

     print(f'{your_output} grdes are {student[your_output]}')

email=input('enter your email: ')
if '@'in email or '.' in email:
     print(f"your email {email} is valid")
else:
     print('invaid email')

sachin=[s*2 for s in range(1,11)]
print(sachin)'''


'''

from datetime import datetime

def show_date_time() -> None:
     print('this is your curent date time ')
     print(datetime.now())

show_date_time()
show_date_time()


from random import choices , sample

lists=['sachin','shirish','girish','sharamik','akash']
for i in range(5):
     new_list=sample(lists,k=2)
     print(new_list)

from operator import itemgetter
for i in range(4):
     sachin=itemgetter(0,2)
print(sachin(lists))'''


'''class Animal:
     
     def __init__(self,name,speed) -> None:
          self.name=name
          self.speed=speed
          self.running=True

     def show_speed(self):
          print(f'{self.name} is very speed \n and car speed is {self.speed}')


class bugati(Animal):

     def info(self):
          pass

car = bugati('bmw', '200kmph')
car.show_speed()


from abc import ABC,abstractclassmethod

class vehical(ABC):
     @abstractclassmethod
     def car(self):
          print(f'the car is very speed')
     @abstractclassmethod
     def car_price(self):
          print('car price is ')

class porche(vehical):
     def car(self):
          print(f'the car is very speed')

     def car_price(self):
          print('car price is ')



          
c=porche()
c.car()
c.car_price()'''


'''studen_list={'bugati':3,
             'meclaren':2,
             'porche':1,
             'BMW':1,
             'ferrari':2,
             'pagani':2,
             'mercedes':2,
             'mustang':5}

your_car_list=[]

for key,values in studen_list.items():
     print(f'{key:10} {values}')

while True:
     your_car=input('enter your car name and this type a s:👉  ')
     if your_car.lower() == 's':
          break

     if your_car in studen_list:
          print(f'this car is a {your_car} and car price is a {studen_list[your_car]} in a M$')
          your_car_list.append(your_car)

     elif studen_list.get(your_car) is  None:
          print(f'{your_car} is not in the list🙃')
print()
print('⭐------------THIS IS YOUR CAR LIST------------⭐')
print()
for cars in your_car_list:
     print(cars)'''


'''menu={'roti':100,
      'naan':200,
      'paneer':400,
      'dal tadaka':400,
      'paratha':300,
      'solkadi':50,
      'papad':20}

orders=[]
print('-----------⭐----------')
name=input('enter your name: ')
print(f'{name} Wlcome our restro🙏')
print()
print('this is our menu')
print()
for key,vlaue in menu.items():
     print(f'{key:10} {vlaue} ')

while True:
     your=input('enter your order or a (q to quite):  ')
     if your.lower() == 'q':
          break

     if your in menu:
          print(f'your order is {your} and price is {menu[your]}')
          orders.append(your)

     elif menu.get(your) is None:
          print(f'{your} not in our list🙃')
print()
print('----------⭐Your list⭐------------')

for food in orders:
     print(food)
print()
total = 0
for food in orders:
     total += menu.get(food)

print(f'total price is {total}')'''



temp1=[12.34,32,43.3,22,12.3]
new_temp=list(map(lambda temp: (temp * 9/5) +32 , temp1))
print(new_temp)

my_list=[12,323,33,344,545,655,54,43,32,43]
sachin=list(filter(lambda grade: grade<=100,my_list))
print(sachin)


menu={'roti':100,
      'naan':200,
      'paneer':400,
      'dal tadaka':400,
      'paratha':300,
      'solkadi':50,
      'papad':20}

order=dict(sorted(menu.items(),key=lambda item:item[0], reverse= True))
print(order)

masti=dict(sorted(menu.items(),key=lambda item:item[1]))
print(masti)

name=input('enter your name: ')
print(f'{name} Welcome our restro🙏')
print()
print('----------⭐-----------')
print()
print('----------This is a Menu-----------')

order_your=[]
for key,values in menu.items():
     print(f'{key:10} {values:.2f}')

print()

while True:
     your_order=input('enter your order or a (q to stop): ')
     if your_order.lower() =='q':
          break

     if your_order in menu:
          print(f'here your order {your_order} and \n your order price is {menu[your_order]}')
          order_your.append(your_order)
     
     elif menu.get(your_order) is None:
          print(f'this {your_order} not available')
print()
print('-----------⭐your order list⭐----------')
for food in order_your:
     print(food)
print()
print('---------⭐your total bil⭐----------')
total_bill=0
for food in order_your:
     total_bill+=menu.get(food)

print(f'this is your total bill {total_bill}')