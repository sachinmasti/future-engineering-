







# # class mclarn:
# #     def __init__(self,name,price):
# #         self.name = name
# #         self.price = price

# #     def __len__(self):
# #         count = 0
# #         for s in self.name:
# #             count = count+1
# #         return count
    
# #     def __call__(self, *args: Any, **kwds: Any) -> Any:
# #         return (f"my car is {self.name} and my car price is {self.price}")
# # dodge = mclarn("bmw" , "500k")
# # print(dodge.name,dodge.price)
# # print(len(dodge))
# # print(dodge())

# from typing import Any

# class sports:
#     def __init__(self,name,country):
#         self.sports_name = name
#         self.country = country

#     def sachin(self):
#         print("the sports is the most beutifull thing in the planet earth")


#     def __call__(self):
#         return (f"this is {self.sports_name} and \n the origin country is {self.country}")

# class cricket(sports):
#     def __init__(self,sports_name,country):
#         sports.__init__(self,sports_name,country)
        

#     def __call__(self):
#         return (f"the {self.sports_name} is the very popular game in the world and \n {self.country} is the origin country of this game")

# sacchi = cricket("cricket","england")
# sacchi.sachin()
# print(sacchi.sports_name,sacchi.country)
# print(sacchi())

class men:
    def __init__(self,name):
        self.name=name

    def info(self):
        print(f"the men name is {self.name}")

class sports:
    def __init__(self,sports_name):
        self.sports_name = sports_name

    def info(self):
        print(f"the sports name is {self.sports_name}")
    
class person(sports,men):
    def __init__(self,name,sports_name):
        self.name = name
        self.sports_name = sports_name

sm = person("sachin","cricket")
print(sm.name,sm.sports_name)
print(sm.info())
print(person.mro())

class vechical:
    def __init__(self,vechical_name,price):
        self.vechical_name = vechical_name
        self.price = price

    def show(self):
        print(f"the car name is {self.vechical_name} \n the car price is {self.price}")

class cars:
    def __init__(self,car_name,car_price):
        self.car_name = car_name
        self.car_price = car_price

    def show(self):
        print(f"the car name is {self.car_name} \n the cost of this car is {self.car_price}")

class bugati(cars,vechical):
    def __init__(self,name,price):
        self.name = name
        self.price = price

    def info(self):
        print(f"the name of {self.name} the price is {self.price}")

s = bugati("mustang","500k$")
print(s.name,s.price)
print(s.info())
print(bugati.mro())