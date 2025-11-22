# def sachin():
#     print("my name is sachin ")
#     print("my fav game is cricket")
 
# if __name__=="__main__":
#     sachin()



# s=[1,2,3,4,5,7,9,8,10]
# sachin=["even" for i in s if i%2==0 ]
# print(sachin)

# for i in  range(1,11):
#     for j in range(1,11):
#         if i%2== 0 and j%2==1:
#             print((i,j))

# [(i,j) for i in range(1,10) if i%2==0 for j in range(1,10) if j%2==1]

#m=[1,2,3,4,5,6,10,3,1,9]
# for i in m:
#     if i%2==0:
#         print("yes🤣")
#     elif i%2==1:
#         print("no😒")
#     else:
#         print("done❤️‍🔥")

# sachin=["yes🤣" if i==0 else "no😒" if i==1 else "done👍" if i==3 else "anything" for i in m ]
# print(sachin)


# s= lambda x,y : 'none' if y==0 else x/y 
# s=(0,4)
# print(s)



# import requests
# url="https://images.unsplash.com/photo-1694547278143-4c83c9d46b1e?w=900&auto=format&fit=\
#  crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHx0b3BpYy1mZWVkfDZ8Ym84alFLVGFFMFl8fGVufDB8fHx8fA%3D%3D"
# response=requests.get(url) 
# with open("unsplash2.jpg","wb") as sachin:
#      sachin.write(response.content)

# url="https://youtu.be/Tl4bQBfOtbg?list=RDCLAK5uy_k0DHSGtf1u4BCuS8huQlheNyJ_BHvvWm4"
# sm=requests.get(url)
# with open("youtube.MPEG-2","wb") as sachin:
#       sachin.write(sm.content)

# import urllib
# dwn_link = 'https://www.youtube.com/watch?v=Tl4bQBfOtbg&pp=ygUMcmFtIHNpeWEgcmFt'

# file_name = 'trial_video.mp4' 
# urllib.request.urlretrieve(dwn_link,file_name)

# import requests
# import time
# def sachin():
#     time.sleep(4)
#     url="https://youtu.be/0Iu5kQi8lns?t=11"
#     response=requests.get(url)
#     open("youtube.mp4","wb").write(response.content)

# sachin()
    

# import plotly.express as px 


# # Creating the Figure instance
# fig = px.line(x=[1,2, 3], y=[1, 2, 3]) 

# # printing the figure instance
# print(fig)
import plotly.graph_objects as go 
import numpy as np 

# Data to be plotted
x = np.outer(np.linspace(-2, 2, 30), np.ones(30)) 
y = x.copy().T 
z = np.cos(x ** 2 + y ** 2) 

# plotting the figure
fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)]) 

fig.show()
