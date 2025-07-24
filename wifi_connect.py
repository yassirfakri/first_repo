from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.keys import Keys
import subprocess
from dataclasses import dataclass
import time

WAITING_TIME = 7  # in seconds


@dataclass
class Auth:
    username: str
    password: str


class ConnectionError(Exception):
    pass


def connect_to_wifi_with_auth(ssid: str, auth: Auth) -> None:
    try:
        print(f"Trying to connect to the Wifi '{ssid}'...")
        # Use the netsh command to connect to the Wi-Fi network on Windows
        subprocess.run(['netsh', 'wlan', 'connect', 'name=', ssid], check=True)

        edge_service = Service("./edgedriver_win64/msedgedriver.exe")
        edge_options = webdriver.EdgeOptions()

        # Handling some Selenium errors
        edge_options.add_argument("--disable-proxy-certificate-handler")
        edge_options.add_argument("--enable-chrome-browser-cloud-management")
        # edge_options.add_argument("--headless")
        edge_options.add_experimental_option(
            'excludeSwitches', ['enable-logging'])

        driver = webdriver.Edge(service=edge_service, options=edge_options)
        driver.get("https://www.google.com")  # Opening a random web page
        main_window_handle = driver.current_window_handle

        # Waiting for the authentification portal to open (eventually)
        time.sleep(WAITING_TIME)

        # Switching to the wifi authentification portal if it popped up
        handles = driver.window_handles
        if len(handles) == 1:
            print("Connection successful: authentification is not necessary.")
        else:
            new_window_handle = [
                handle for handle in handles if handle != main_window_handle][0]
            driver.switch_to.window(new_window_handle)
            handle_authentification_portal(driver=driver, auth=auth)
    except Exception as e:
        raise ConnectionError(f"Error connecting to {ssid}: {e}")
    finally:
        driver.quit()  # Close the browser window


def handle_authentification_portal(driver: webdriver.Edge, auth: Auth) -> None:
    portal_url = driver.current_url
    if '10.254.0.254:1000/fgtauth' in portal_url:
        print("Handling firewall authentification portal:", portal_url)
        username_field = driver.find_element("name", "username")
        password_field = driver.find_element("name", "password")
        username_field.send_keys(auth.username)
        password_field.send_keys(auth.password, Keys.RETURN)
        print("Authentication successful.")
    else:
        raise ConnectionError(f"Unknown portal URL: {portal_url}")


if __name__ == '__main__':
    auth = Auth(username='MEF_614', password='eUCctsFJ')
    connect_to_wifi_with_auth(ssid='WifiCity', auth=auth)
