from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.common.by import By

# What you enter here will be searched for in
# Google Images
query = "우회전 신호등"
 
# Creating a webdriver instance
driver = webdriver.Chrome()
 
# Maximize the screen
driver.maximize_window()
 
# Open Google Images in the browser
driver.get('https://images.google.com/')
 
# Finding the search box
box = driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
# box = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')

# Type the search query in the search box
box.send_keys(query)
 
# Pressing enter
box.send_keys(Keys.ENTER)
 
# Function for scrolling to the bottom of Google
# Images results
def scroll_to_bottom():
 
    last_height = driver.execute_script('\
    return document.body.scrollHeight')
 
    while True:
        driver.execute_script('\
        window.scrollTo(0,document.body.scrollHeight)')
 
        # waiting for the results to load
        # Increase the sleep time if your internet is slow
        time.sleep(3)
 
        new_height = driver.execute_script('\
        return document.body.scrollHeight')
 
        # click on "Show more results" (if exists)
        try:
            driver.find_element(By.CSS_SELECTOR, ".YstHxe input").click()
 
            # waiting for the results to load
            # Increase the sleep time if your internet is slow
            time.sleep(3)
 
        except:
            pass
 
        # checking if we have reached the bottom of the page
        if new_height == last_height:
            break
 
        last_height = new_height
 
 
# Calling the function
 
# NOTE: If you only want to capture a few images,
# there is no need to use the scroll_to_bottom() function.
scroll_to_bottom()
 
 
# Loop to capture and save each image
for i in range(1, 50):
   
    # range(1, 50) will capture images 1 to 49 of the search results
    # You can change the range as per your need.
    try:
 
      # XPath of each image
        img = driver.find_element(By.XPATH, 
            '//*[@id="islrg"]/div[1]/div[' +
          str(i) + ']/a[1]/div[1]/img')
 
        # Enter the location of folder in which
        # the images will be saved
        img.screenshot('./trafficlight/' + 
                       query + ' (' + str(i) + ').png')
        # Each new screenshot will automatically
        # have its name updated
 
        # Just to avoid unwanted errors
        time.sleep(0.2)
 
    except:
         
        # if we can't find the XPath of an image,
        # we skip to the next image
        continue
 
# Finally, we close the driver
driver.close()