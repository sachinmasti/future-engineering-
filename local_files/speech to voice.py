
# import pyttsx3
# engine = pyttsx3.init()
# engine.say("kamlesh")
# engine.runAndWait()



import pyttsx3
import keyboard # you need to install this library using pip install keyboard
engine = pyttsx3.init()
running = True # this is the variable that controls the loop
while running: # this is the while loop
  engine.say("sakshi")
  engine.runAndWait()
  if keyboard.is_pressed('q'): 
    running = False # this will end the loop


lst = ["a", "b", "c", "d", "e"]
dct = dict.fromkeys(lst)
print(dct)
# Output: {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1}

key=["sachin","masti"]
values=[1,2]
print(dict(zip(key,values)))

