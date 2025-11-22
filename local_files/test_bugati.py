# s=lambda x,y :x/y
# s=100,200
# print(s)


# import random

# def game():
#     number_to_guess = random.randint(1, 10)
#     guess = None
#     while guess != number_to_guess:
#         guess = int(input("Guess a number between 1 and 10: "))
#         if guess < number_to_guess:
#             print("Too low!")
#         elif guess > number_to_guess:
#             print("Too high!")
#     print("Congratulations! You guessed the number.")

# if __name__ == "__main__":
#     game()


# def cube(x):
#     return x*x*x

# s=[10,3,4,5,6,7,8]
# newl=list(map(cube,s))
# print(newl)

# def filter_func(a):
#     return a>5

# newli=list(filter(filter_func,s))
# print(newli)


# from functools import reduce
# m=[1,2,3,4,5,6,7]
# def mysum(x,y):
#     return x+y
# sachin=reduce(mysum,m)
# print(sachin)




# def sachin(n):
#     if n==1:
#         return 1
#     else :
#         (sachin * )

# def count_digit(start, end, digit):
#     count = 0
#     for num in range(start, end+1):
#         temp = num
#         while temp > 0:
#             if temp % 10 == digit:
#                 count += 1
#             temp //= 10
#     return count

# start = int(input("Enter start number: "))
# end = int(input("Enter end number: "))
# digit = int(input("Enter digit to count: "))

# print(f"The digit {digit} appears {count_digit(start, end, digit)} times between {start} and {end}.")



# import random

# def roll_dice():
#     return random.randint(1, 6)

# result = roll_dice()
# print(f"The result of the dice roll is {result}.")


# def celsius_to_fahrenheit(celsius):
#     return celsius * 9/5 + 32

# # Test the function
# celsius = int(input("enetr your tempreture : "))
# print(f"{celsius} degrees Celsius is equal to {celsius_to_fahrenheit(celsius)} degrees Fahrenheit.")

# def sum_of_odds(numbers):
#     return sum(num for num in numbers if num % 2 != 0)

# # Test the function
# numbers = [1, 2, 3, 4, 5, 6]
# print(f"The sum of odd numbers in the list is {sum_of_odds(numbers)}.")

# s=[1,2,3,4,5,6,7,8]
# sum_of_odds(s)

class sachin:
    def __init__(self,name,sports):
        self.name = name
        self.sports = sports

class masti(sachin):
    def __init__(self,name,sports,age):
        super().__init__(name,sports)
        self.age = age

sm=masti("sachin","cricket","22")
print(sm.name)
print(sm.sports)
print(sm.age)

class bugati:
    def __init__(self,car_name,car_price,):
        self.name=car_name
        self.price = car_price

class cars(bugati):
    def __init__(self, car_name, car_price,car_speed):
        super().__init__(car_name, car_price)
        self.speed = car_speed

mustang = cars("porche","3million","220mph")
print(mustang.name)
print(mustang.price)
print(mustang.speed)

class car:
    def __init__(self,name,price,country):
        self.name=name
        self.price=price
        self.country=country

class mycar(car):
    def __init__(self, name, price, country,speed):
        super().__init__(name, price, country)
        self.speed=speed

bmw=mycar("mustang","500k","use","280kmph")
print(bmw.name)
print(bmw.price)
print(bmw.country)
print(bmw.speed)