# car=["buagti","ferrari","mclaren","bmw"]
# for i,sm in enumerate(car):
#     print(i+1,sm)


# ab=int(input("enter your column :👉 "))
# cd=int(input("enter your rows :👉 "))
# your=input("enter your input :👉 ")
# for i in range(ab):
#     for s in range(cd):
#         print(your,end="")
#     print()

# def do_that():
#     print("do that😍")

# def do_this():
#     print("do this 👽")


# while True:
#     sachin=input("enter your choise :👉 ")
#     match sachin:
#         case "that":
#             do_that()
#         case "this":
#             do_this()
#         case _:
#             print("invalid input🙃")
#             break

list1=[1,2,3,4,5,6,7,8,9,0]
list2=[1,2,3,4,5,6,7,89,0,]

def original(l1,l2):
    return sorted(set(l1+l2))

c=original(list1,list2)
print(c)

my_list=["sachin","girish","sachin","shirish"]
my_list1=["sachin","akash"]
d=original(my_list,my_list1)
print(d)

masti=range(1,21)
def sacchi(num):
    for i in range(2,num):
        if (num%i)==1:
            return False
        return True
    
number=list(filter(sacchi,masti))
print(number)                 