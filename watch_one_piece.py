from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from typing import Any
import pdb as debugger
from seleniumbase import Driver

pdb = debugger.set_trace

ANIME = "One Piece"
EPISODE = 1000
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def click_using_js(driver: Any, element: Any) -> None:
    # Scroll into view if necessary
    driver.execute_script("arguments[0].scrollIntoView(true);", element)

    # Click on the element using JavaScript
    driver.execute_script("arguments[0].click();", element)


if __name__ == '__main__':
    driver = Driver(uc=True, ad_block_on=True)  # Undetected selenium bot
    driver.get("https://anix.to")

    search_box = driver.find_element("name", "keyword")
    search_box.send_keys(ANIME, Keys.RETURN)

    # anime = driver.find_element(By.XPATH, f"//a[text()='{ANIME.upper()}']")
    anime = driver.find_element(
        By.XPATH, f"//a[translate(text(), '{ALPHA}', '{ALPHA.lower()}')='{ANIME.lower()}']")
    click_using_js(driver=driver, element=anime)

    form_element = driver.find_element(By.CLASS_NAME, "ep-search")
    ep_input_element = form_element.find_element(By.CLASS_NAME, "form-control")
    ep_input_element.send_keys(EPISODE, Keys.RETURN)

    # server = driver.find_element(
    #     By.XPATH, f"//div[contains(@class, 'server') and @data-sv-id='35' and @data-cmid='anix_ov8']")
    # click_using_js(driver=driver, element=server)
    pdb()
    driver.quit()
