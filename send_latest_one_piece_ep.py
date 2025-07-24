import smtplib
import ssl
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from seleniumbase import Driver
from watch_one_piece import click_using_js
from dataclasses import dataclass
import pdb as debugger
from bs4 import BeautifulSoup


pdb = debugger.set_trace


@dataclass
class OnePiece:
    episode: int
    link: str


def generate_one_piece_mail(sender: str, receiver: str, episode_element:
                            OnePiece) -> MIMEMultipart:
    message = MIMEMultipart()
    message["Subject"] = "Latest One Piece episode"
    message["From"] = sender
    message["To"] = receiver

    html = f"""\
    <html>
    <body>
        <p>Hello,<br>
        <br>
        The latest episode of One Piece is {episode_element.episode}, and is available at:
        <a href="{episode_element.link}">Watch One Piece</a><br>
        <br>Have fun !<br>
        <br><i>Sent by Python<i>
        <br></p>
    </body>
    </html>
    """
    message.attach(MIMEText(html, "html"))
    return message


def get_latest_op_episode() -> OnePiece:
    driver = Driver(uc=True, headless=True)  # Undetected selenium bot
    driver.get("https://anix.to")

    search_box = driver.find_element("name", "keyword")
    search_box.send_keys('One Piece', Keys.RETURN)

    anime = driver.find_element(By.XPATH, f"//a[text()='ONE PIECE']")
    click_using_js(driver=driver, element=anime)

    html_content = driver.page_source
    soup = BeautifulSoup(html_content, "html.parser")

    div_text = soup.find(
        "div", {"class": "dropdown-menu range-options"}).get_text()
    episode_ranges = div_text.strip().split(" ")
    latest_episode = int(episode_ranges[-1].split("-")[-1])

    form_element = driver.find_element(By.CLASS_NAME, "ep-search")
    ep_input_element = form_element.find_element(By.CLASS_NAME, "form-control")
    ep_input_element.send_keys(latest_episode, Keys.RETURN)

    try:
        WebDriverWait(driver, 10).until(EC.url_contains(str(latest_episode)))
        episode_link = driver.current_url
    except TimeoutException:
        raise Exception(
            f"Episode link does not contain episode {latest_episode}")
    finally:
        driver.quit()

    return OnePiece(latest_episode, episode_link)


if __name__ == '__main__':
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "yassir.dev.python@gmail.com"
    receiver_email = "fatihazohour@gmail.com"
    password = "mtbjzysqbkklmyvk"  # os.environ.get('dev_mail_password')

    print("Fetching latest One Piece episode...")
    ep_element = get_latest_op_episode()
    print("Latest One Piece episode has been fetched successfully")

    message = generate_one_piece_mail(
        sender=sender_email, receiver=receiver_email, episode_element=ep_element)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        print("Sending mail...")
        server.sendmail(
            sender_email, receiver_email, message.as_string()
        )
        print("Mail has been sent Successfully")
