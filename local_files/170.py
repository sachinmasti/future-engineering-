# import asyncio
# import requests

# async def sachin():

#     await asyncio.sleep(5)
#     print("this is here👍")
#     URL = "https://images.unsplash.com/photo-1703818770847-0f2948e0611b?q=80&w=3174&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
#     response = requests.get(URL)
#     open("instagram.ico", "wb").write(response.content)

# async def masti():
#     await asyncio.sleep(5)
#     print("this work is done👽")
#     URL = "https://images.unsplash.com/photo-1675326570919-946d728e9a25?q=80&w=2424&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D.jpg"
#     response = requests.get(URL)
#     open("instagram2.jpg", "wb").write(response.content)

# async def main():
#     t=await asyncio.gather(
#         sachin(),
#         masti()
#     )
#     print(t)
#     #  await sachin(),
#     #  await masti()
# asyncio.run(main())

# import asyncio
# import requests

# async def bugati():
#     await asyncio.sleep(3)
#     print("this work is done👽")
#     url="https://unsplash.com/photos/a-house-sitting-on-top-of-a-lush-green-hillside-cDBZq75p1FI"
#     responce=requests.get(url)
#     open("sachin845.jpg","wb").write(responce.content)

# async def ferrari():
#     await asyncio.sleep(2)
#     print("the work okay❤️‍🔥")
#     url ="https://unsplash.com/photos/a-pie-sitting-on-top-of-a-white-plate-next-to-a-fork-q8__sinkeco"
#     responce=requests.get(url)
#     open("sachinmasti845.io","wb").write (responce.content)

# async def main():
#     # await bugati()
#     # await ferrari()

#     sm= await asyncio.gather(
#         bugati(),
#         ferrari()
#     )
# asyncio.run(main())
from win10toast import ToastNotifier

# One-time initialization
toaster = ToastNotifier()

# Show notification whenever needed
toaster.show_toast("Notification!", "drink water", threaded=True,
                   icon_path=None, duration=3)  # 3 seconds

# To check if any notifications are active,
# use `toaster.notification_active()`
import time
interval=5
while True:
 toaster.notification_active()
time.sleep(interval)