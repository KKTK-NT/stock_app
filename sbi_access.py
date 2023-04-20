from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from get_data import get_table

def get_sbi_holdings(username, password):
    url = "https://www.sbisec.co.jp/ETGate"  # SBI証券のURL
    
    # ブラウザを開く
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    service = Service(executable_path='./chromedriver')
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)

    # ユーザー名とパスワードを入力
    username_input = driver.find_element(by=By.NAME, value='user_id')
    username_input.send_keys(username)
    password_input = driver.find_element(by=By.NAME, value='user_password')
    password_input.send_keys(password)

    # ログインボタンをクリック
    login_button = driver.find_element(by=By.NAME, value='ACT_login')
    login_button.click()

    # 保有銘柄ページに移動
    time.sleep(2)  # 必要に応じて調整
    
    return get_table(driver)
