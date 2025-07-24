from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from typing import Any
import time
import pdb as debugger
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from watch_one_piece import click_using_js


pdb = debugger.set_trace

edge_service = Service("./edgedriver_win64/msedgedriver.exe")
edge_options = webdriver.EdgeOptions()

# Handling some Selenium errors
edge_options.add_argument("--disable-proxy-certificate-handler")
edge_options.add_argument("--enable-chrome-browser-cloud-management")
edge_options.add_experimental_option(
    'excludeSwitches', ['enable-logging'])

driver = webdriver.Edge(service=edge_service, options=edge_options)
driver.get("http://192.168.1.1")
username_field = driver.find_element("name", "Login_Name")
password_field = driver.find_element("name", "Login_Pwd")
username_field.send_keys("admin")
password_field.send_keys("TNCAPA2224DD2", Keys.RETURN)
time.sleep(2)
maintenance_elt = driver.find_element(
    By.XPATH, "a[onclick*='top.main.location=&quot;../maintenance/tools_admin.htm&quot;']")

click_using_js(driver, maintenance_elt)
pdb()
