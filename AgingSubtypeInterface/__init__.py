import pandas as pd 
import numpy as np 
import pickle 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from fpdf import FPDF
#from wand.image import Image as WImage 

def create_report():
    class PDF(FPDF):
        def lines(self):
            self.set_fill_color(145.0, 204.0, 241.0) # color for outer rectangle
            self.rect(5.0, 5.0, 200.0,287.0,'DF')
            self.set_fill_color(255, 255, 255) # color for inner rectangle
            self.rect(8.0, 8.0, 194.0,282.0,'FD')

        def imagex(self):
            self.set_xy(8.5,10.5)
            self.image('/home/vikram/Projects/Sustain-FindSubtype-Test/emc.jpg',
            link='', type='', w=1586/40, h=1920/120)
            self.set_xy(180.0,10.5)
            self.image('/home/vikram/Projects/Sustain-FindSubtype-Test/europond.jpg',  
            link='', type='', w=1586/80, h=1920/80)
            self.set_xy(88.0,8.5)
            self.image('/home/vikram/Projects/Sustain-FindSubtype-Test/sustain.jpg',  
            link='', type='', w=34, h=20)
            self.set_xy(15.0,105)
            self.image('/home/vikram/Projects/Sustain-FindSubtype-Test/subtype_1.png',  
            link='', type='', w=180, h=50)
            self.set_xy(105,47)
            self.image('/home/vikram/Projects/Sustain-FindSubtype-Test/prob.png',  
            link='', type='', w=60, h=45)

        def texts(self):
            self.set_xy(10.0,45.0)    
            self.set_text_color(0, 0, 0)
            self.set_font('Times', 'B', 14)
            self.cell(0,0,'Subject\'s Result Summary:')
            self.set_font('Times', '', 12)
            self.set_xy(10.0,65.0)   
            self.cell(0,0,'Sex: Male')
            self.set_xy(10.0,70.0)
            self.cell(0,0,'Age: 73')
            self.set_xy(10.0,75.0)
            self.cell(0,0,'Most likely subtype: Early cortical atrophy subtype')
            self.set_xy(10.0,80.0)
            self.cell(0,0,'Model stage: 15 of 33')
            self.set_xy(25.0,100.0)
            self.multi_cell(0,0,'The figure below shows the typical brain atrophy pattern in the subtype the subject is most likely in.')

    pdf = PDF()#pdf object
    pdf.add_page()
    pdf.lines()
    pdf.imagex()
    pdf.texts()
    pdf.output('Report.pdf','F')

    return 

def view_report():
    #img = WImage(filename='Report.pdf')
    #img
    return 

def email_report(your_emailid, password, receiver):

    msg = MIMEMultipart()
    
    msg['From'] = your_emailid
    msg['To'] = receiver 
    msg['Subject'] = "Report: Brain Aging subtype"
    body = "This e-mail message contains the automatic report generated \
        by the web application for subtyping aging brain."
    
    msg.attach(MIMEText(body, 'plain'))
    
    # open the file to be sent 
    filename = "Report.pdf"
    attachment = open("Report.pdf", "rb")
    
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    
    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    try:
        smtpObj = smtplib.SMTP('smtp.gmail.com',587)
        smtpObj.ehlo()
        smtpObj.starttls()
        smtpObj.ehlo()
        smtpObj.login(your_emailid,password)
        smtpObj.sendmail(your_emailid,receiver,msg.as_string())
        smtpObj.quit()    
        print ("Successfully sent email")
    except smtplib.SMTPException:
        print ("Error: unable to send email")

    return

def read_trained_model():

    return

def estimate_aging_subtype():

    return 