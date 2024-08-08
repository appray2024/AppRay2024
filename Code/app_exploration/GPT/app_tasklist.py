
emails_passwords = {
    "com.woolworths": {
        "email1": "xxx@gmail",
        "password1": "xxx",
        # for registatrion
        "email2": "xxx@gmail",
        "password2": "xxx",
    }
}

app2tasks = {
        "com.woolworths": {
                "appName": "Woolworths",
                "tasks": [  
                        "9-register or create a new account using email [{}] Password [{}]. Can use Google for signup if email registration is not allowed. Fill other required information by yourself".format(emails_passwords["com.woolworths"]["email2"], emails_passwords["com.woolworths"]["password2"]),
                        "10-log in the account using Account: {} Password: {}".format(emails_passwords["com.woolworths"]["email1"], emails_passwords["com.woolworths"]["password1"]),
                        "3-Go to setting page, go through all notification related pages. no need to turn off options.",
                        "4-Go to setting page, go through all privacy related setting. no need to turn off options.",
                        "5-Check if we can subscribe to premium account, if so, read through all contents on the subscription page and proceed to payment page", 
                        "7-Go shopping, select any product you like with proper attributes (like size, color), add to cart, proceed to checkout",
                        "8-Sign out the account. You may need to scroll in the setting page to find the sign out option.",
                ],
        },        

}