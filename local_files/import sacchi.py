



# class bugati:
#     def __init__(self,car):
#         self.car=car

#     def bmw(self):
#         print(f"the value is {self.car}")

#     @ property
#     def porche(self):
#         return 10 * self.car
    
#     @porche.setter
#     def porche(self,new_car):
#         self.car= new_car + 10

# obj=bugati(20)
# obj.bmw()

# class ferrari:
#     def __init__(self,sm):
#         self.sm = sm
#     def info(self):
#         print(f"the value is {self.sm}") 

#     @property
#     def sachin(self):
#         return 10 * self.sm
    
#     @sachin.setter
#     def sachin(self,number):
#         self.sm = number + 10
# obj=ferrari(100)
# obj.info()

# class hyper_car:
#     car_country = "italy"
#     car_count = 0
#     def __init__(self,car_name):
#         self.car_name=car_name
#         self.car_price = "10 million"
#         hyper_car.car_count +=1

#     def car_info(self):
#         print(f"the car no {self.car_count} the car country is  {self.car_country} the car name is {self.car_name} and the car price is {self.car_price}\n")


# sm=hyper_car("ferrari")
# sm.car_info()

# sm1=hyper_car("bugati")
# sm1.car_country = "france"
# sm1.car_price = "12 million"
# sm1.car_info()


# class porche:
#     def __init__(self,num):
#         self.num=num

#     def info(self):
#         print(f"the value is {self.num}")

#     @property
#     def ferrari(self):
#         return 10 * self.num
    
#     @ferrari.setter
#     def ferrari(self,new_num):
#         self.num = new_num + 100
#     @staticmethod
#     def sachin(a,b):
#         return a + b
# sm2=porche(200)
# sm2.ferrari = 300
# print(sm2.ferrari)
# sm2.info()
# print(porche.sachin(200,100))


# class sports:
#     sports_country="india"
#     sports_count = 0

#     def __init__(self,sports_name):
#         self.sports_name = sports_name
#         self.player_number = 11
#         sports.sports_count +=1

#     def info(self):
#         print(f"game number is {self.sports_count} the sports name is {self.sports_name} the plyers in game {self.player_number} game origin country is {self.sports_country}")

# masti=sports("hockey")
# masti.info()

# masti1=sports("cricket")
# masti1.sports_country = "england"
# masti1.info()

# class cricket:
#     def __init__(self,number):
#         self.number = number


#     def info(self):
#         print(f"thr number is {self.number}")

#     @property
#     def mustang(self):
#         return 10 * self.number

#     @mustang.setter
#     def mustang(self,new_number):
#         self.number = new_number + 100

#     @staticmethod
#     def masti(a,b):
#         return a + b
    
# bmw=cricket(200)
# bmw.info()
# bmw.mustang=100
# print(bmw.mustang)
# bmw.info()
# print(cricket.masti(200,100))

# class football:
#     game_name = "football"
#     def details(self):
#         print(f"the game come frome {self.name},and the game is {self.game_name}")
    
#     @classmethod
#     def change_name(cls,new_game):
#         cls.game_name = new_game

# sachin = football()
# sachin.name ="england"
# sachin.details()

# sachin.change_name("cricket")
# sachin.details()

class cars:
    car_name = "bugati"
    car_no = 0
    car_price = "1 million "
    def __init__(self,car_country):
        self.car_country = car_country
        self.car_name
        cars.car_no +=1

    def info(self):
        print(f"the car no is {self.car_no} \nand the car name is {self.car_name
        } and the car price is {self.car_price} and the  car come from {self.car_country}")

sachin = cars("bugati")
sachin.car_country = "france"
sachin.car_price = "4 million "
sachin.info()




# class ferrari:
#     def __init__(self,name,age):
#         self.name = name
#         self.age = age
#     @classmethod
#     def cahnge(cls,sachin):
#         return cls(sachin.split("-")[0],int(sachin.split("-")[1]))
    

# sm = ferrari("sachin  ",100000)
# print(sm.name)
# print(sm.age)

# cahnge = "sachin - 1000"
# sm1=ferrari.cahnge(cahnge)
# print(sm1.age)
# print(sm1.name)

# class cricket:
#     car_count = 0
#     def __init__(self,name,price):
#         self.car = name 
#         self.car_price = price
#         cricket.car_count +=1

#     @classmethod
#     def sachin(cls , masti):
#         return cls(masti.split("-")[0],int(sachin.split("-")[1]))
    


# sm = cricket("bugati",30000)
# print(sm.car)
# print(sm.car_count)
# print(sm.car_price)

# sachin ="mustang - 10000"
# sm2 = cricket.sachin(sachin)
# print(sm2.car_count)
# print(sm2.car)
# print(sm2.car_price)

class cars:
    def __init__(self,car_name,car_price):
        self.car_name = car_name
        self.car_price = car_price

s = cars ("bugati", 3)
print(s.__dict__)

class music:
    m_count = 0

    def __init__(self,name,album):
        self.singer = name
        self.song = album
        music.m_count +=1

    @classmethod
    def new (sm , sachin ):
        return sm(sachin.split("-")[0],sachin.split("-")[1])
    
sa = music("sidhu moosewala","same beaf")
print(sa.m_count)
print(sa.singer)
print(sa.song)
print(sa.__dict__)

new = "dua lipa - no lie"
sm = music.new(new)
print(sm.m_count)
print(sm.singer)
print(sm.song)
print(sm.__dict__)

s ={
    "name" : "nan",
    "sports" : "nan",
    "country" : "nan"
}

m ={
    "name" : input("enetr your name : "),
    "sports" : input("enetr your game : "),
    "country" : input("enetr your home country : ")
}

new_list = s|m
print(new_list)