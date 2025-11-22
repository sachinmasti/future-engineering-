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

# name:str="sachin masti"
# print(f"{name = }")

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

# a=[1,2,3,4,5,6,7,8,9]
# b=[1,2,3,4,5,6,7,8,9]
# def list_sorted(arry1,arry2):
#     return sorted(set(arry1+arry2))
# c=list_sorted(a,b)
# print(c)

num=list(range(1,20))

def is_primes(nums):
    for y in range(2,nums):
        if(nums%y)==2:
            return False
        return True
primes=list(filter(is_primes,num))
print(primes)        

list1=['sachin'
      'masti',
      'india'
      'girish',
      'shirish',
      22]
first, last, *_, age = list1
print(f" {first}  {last} is {age} year old")

sachin=[12,23,34,45,56,67,78,89,90]
masti=[90,89,78,67,56,45,34,23,12,]
def list_sorted(l1,l2):
    return sorted(set(l1+l2))
sm=list_sorted(sachin,masti)
print(sm)


'''name=input("enter your name : ")
new_name= name or "n/a"
print(new_name)'''

my_list=[]
while True:
    your_input=input('enter your list : ')
    if 'sachin' in your_input:
        print("sachin  you are a great")
        break
    else:
        print('countinoue')
        my_list.append(your_input)
print()
print(f'this is your list \n {my_list}')
lenth={name:len(name) for name in my_list}
print(lenth)

new_list=['sachin','girish','akash','shramik','shirish','prakash']
new_list1=['pramod','soham','suraj','sachin','prakash']
def sort_names(list1,list2):
    return sorted(set(list1+list2))
new_out=sort_names(new_list,new_list1)
print(new_out)
for i,things in enumerate (new_out):
    print(i+1,things.title())