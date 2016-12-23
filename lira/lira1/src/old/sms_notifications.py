import smtplib

# Use sms gateway provided by mobile carrier:
# at&t:     number@mms.att.net
# t-mobile: number@tmomail.net
# verizon:  number@vtext.com
# sprint:   number@page.nextel.com

# Establish a secure session with gmail's outgoing SMTP server using your gmail account
sender = "DankEragorn"
phone = "8647109821"
def send_sms(message):
    server = smtplib.SMTP( "smtp.gmail.com", 587 )

    server.starttls()

    server.login( 'awesomeninja777@gmail.com', 'kiritokun' )

    # Send text message through SMS gateway of destination number
    server.sendmail(sender, '%s@mms.att.net' % phone, message)
