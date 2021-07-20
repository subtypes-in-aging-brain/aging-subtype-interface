import pandas as pd 
import numpy as np 
import pickle 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from fpdf import FPDF
import wget
from pdf2image import convert_from_path
import cv2 as cv 
from google.colab.patches import cv2_imshow # for image display
from skimage import io
from matplotlib import pyplot as plt

def view_report():
    pages=convert_from_path("Report.pdf")    
    for page in pages:
      page.save("Report.png","png")
    I=io.imread('Report.png')
    width = int(I.shape[1] * 0.5)
    height = int(I.shape[0] * 0.5)
    dim = (width, height)
    I=cv.resize(I,dim)
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)
    cv2_imshow(I)

    return

def create_report(SubjectID, prediction):
    probabilities = prediction[4][0,:]
    stage = prediction[2][0,0]
    subtype=np.argmax(probabilities)
    subtypes=['Early cortical atrophy', 
        'Early ventricular enlargement']
    plt.bar(subtypes, probabilities, color = '#feb24c', width=0.4)
    plt.title('Probabilities of the subject being \n in the two aging subtypes')
    plt.ylim(0,1)
    plt.savefig('Probabilities.png')
    plt.close()
    class PDF(FPDF):
        def lines(self):
            self.set_fill_color(145.0, 204.0, 241.0) # color for outer rectangle
            self.rect(5.0, 5.0, 200.0,287.0,'DF')
            self.set_fill_color(255, 255, 255) # color for inner rectangle
            self.rect(8.0, 8.0, 194.0,282.0,'FD')

        def imagex(self,subtype):
            self.set_xy(8.5,10.5)
            self.image(wget.download('https://raw.githubusercontent.com/subtypes-in-aging-brain/aging-subtype-interface/main/data/emc.jpg'),
            link='', type='', w=1586/40, h=1920/120)
            self.set_xy(180.0,10.5)
            self.image(wget.download('https://raw.githubusercontent.com/subtypes-in-aging-brain/aging-subtype-interface/main/data/europond.jpg'),  
            link='', type='', w=1586/80, h=1920/80)
            self.set_xy(88.0,8.5)
            self.image(wget.download('https://raw.githubusercontent.com/subtypes-in-aging-brain/aging-subtype-interface/main/data/sustain.jpg'),  
            link='', type='', w=34, h=20)
            self.set_xy(15.0,105)
            if subtype==0:
                self.image(wget.download('https://raw.githubusercontent.com/subtypes-in-aging-brain/aging-subtype-interface/main/data/subtype_1.png'),
                link='', type='', w=180, h=50)
            else:
                self.image(wget.download('https://raw.githubusercontent.com/subtypes-in-aging-brain/aging-subtype-interface/main/data/subtype_2.png'),  
                link='', type='', w=180, h=50)
            self.set_xy(105,47)
            self.image('Probabilities.png',  
            link='', type='', w=60, h=45)

        def texts(self, SubjectID, stage, subtypes, subtype):
            
            self.set_xy(10.0,45.0)    
            self.set_text_color(0, 0, 0)
            self.set_font('Times', 'B', 14)
            self.cell(0,0,'Subject\'s Result Summary:')
            self.set_font('Times', '', 12)
            self.set_xy(10.0,65.0)   
            self.cell(0,0,'Subject ID: ' + SubjectID)
            #self.set_xy(10.0,70.0)
            #self.cell(0,0,'Age: 73')
            self.set_xy(10.0,75.0)
            self.cell(0,0,'Most likely subtype: '+subtypes[subtype]+ ' subtype')
            self.set_xy(10.0,80.0)
            self.cell(0,0,'Model stage: ' + str(int(stage)) +' of 33')
            self.set_xy(25.0,100.0)
            self.multi_cell(0,0,'The figure below shows the typical brain atrophy pattern in the subtype the subject is most likely in.')

    pdf = PDF()#pdf object
    pdf.add_page()
    pdf.lines()
    pdf.imagex(subtype)
    pdf.texts(SubjectID, stage, subtypes, subtype)
    pdf.output('Report.pdf','F')
    view_report()
    return 

def email_report(your_emailid, password, receiver):

    msg = MIMEMultipart()
    
    msg['From'] = your_emailid
    msg['To'] = receiver 
    msg['Subject'] = "Report: Brain Aging subtype"
    body = "This e-mail message contains the automatic report generated by the web application for Subtyping Aging Brain."
    
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

    smtpObj = smtplib.SMTP('smtp.gmail.com',587)
    smtpObj.ehlo()
    smtpObj.starttls()
    smtpObj.ehlo()
    smtpObj.login(your_emailid,password)
    smtpObj.sendmail(your_emailid,receiver,msg.as_string())
    smtpObj.quit()    
    print ("Successfully sent email")

    return