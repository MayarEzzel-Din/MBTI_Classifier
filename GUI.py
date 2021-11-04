import tkinter.font as tkFont
from tkinter import *
import tkinter as tk
from tkinter import ttk
from Training import *

form = tk.Tk()
#form.geometry("400x300")
#form['background']='#95C8D8'
#form.title('Validation')
form.geometry("1920x1080")
form['background'] = '#FFFFFF'
form.title('Personality Traits Detection')
fontStyle = tkFont.Font(family="Segoe UI", size=26)

#posts_lbl = tk.Label(form,text = "Enter The Posts")
#posts_lbl.place(x=10, y=20)
label = tk.Label(form, text="MBTI Personality Traits Detection", fg='#000E90', bg='#FFFFFF', font=("Segoe UI", 26))
label.place(x=490, y=104)

lbl_posts = tk.Label(form, text="Enter Posts", fg='#000E90', bg='#FFFFFF', font=("Segoe UI", 16))
lbl_posts.place(x=164, y=246)

posts_txt = Entry(form, font=("Segoe UI", 16), w=100)
posts_txt.place(x=164, y=308)

#posts_txt = Entry(form)
#posts_txt.place(x=130, y=20)

lbl_result = tk.Label(form, text="Personality Type", fg='#000E90', bg='#FFFFFF', font=("Segoe UI", 16))
lbl_result.place(x=164, y=484)

result_txt = Entry(form, font=("Segoe UI", 16))
result_txt.place(x=164, y=534)

#result_lbl = tk.Label(form,text = "The Result is")
#result_lbl.place(x=10,y=150)
#result_txt=Entry(form)
#result_txt.place(x=130,y=150)

lbl_job = tk.Label(form, text="Job Reccomendation", fg='#000E90', bg='#FFFFFF', font=("Segoe UI", 16))
lbl_job.place(x=1064, y=484)
jobTXT = Entry(form, font=("Segoe UI", 16))
jobTXT.place(x=1064, y=534, height=100)

#jobTXT = Entry(form, font=("Segoe UI", 16))
#jobTXT.place(x=130, y=250)

def call():
    posts = posts_txt.get()
    jobs = []
    result, jobs = validate(posts)
    result_txt.insert(4, result)
    for job in jobs:
        jobTXT.insert(END, job + ' ')

#def jobRecommendation():


Test = Button(form, text="Show Results",command=call, bg='#000E90', fg='#FFFFFF', width=55, height=1, font=("Segoe UI", 14))
Test.place(x=469, y=388)


form.mainloop()
