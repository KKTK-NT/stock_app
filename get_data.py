import os
import time
import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

def format_data(df_data, category, fund):
    # 必要な列のみ抽出
    df_data = df_data.loc[:, ['評価額', '損益', '損益（％）']]

    df_data['カテゴリー'] = category
    if fund != '':
        df_data['ファンド名'] = fund

    return df_data

def export_data(df_result, name):
    df_result.to_csv(name + '_result_{0:%Y%m%d}.txt'.format(datetime.date.today()))

def get_table_ja(driver):
    # 遷移するまで待つ
    time.sleep(0.5)

    # ポートフォリオの画面に遷移
    driver.find_element(by=By.XPATH, value='//*[@id="link02M"]/ul/li[1]/a/img').click()

    # 文字コードをUTF-8に変換
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupでパース
    soup = BeautifulSoup(html, "html.parser")
    symbols = []
    names = []
    
    # 株式
    table_data = soup.find_all("table", bgcolor="#9fbf99", cellpadding="4", cellspacing="1", width="100%") 
    # 株式（現物/特定預り）
    df_stock_specific = pd.read_html(str(table_data), header=0)[0]
    for row in range(len(df_stock_specific)):
        symbol = df_stock_specific.iloc[row,:]["銘柄（コード）"].split()[0]
        name = df_stock_specific.iloc[row,:]["銘柄（コード）"].split()[1]
        symbols.append(symbol)
        names.append(name)
    df_stock_specific = format_data(df_stock_specific, '株式（現物/特定預り）', '上場ＴＰＸ')

    # 株式（現物/NISA預り）
    df_stock_fund_nisa = pd.read_html(str(table_data), header=0)[1]
    for row in range(len(df_stock_fund_nisa)):
        symbol = df_stock_fund_nisa.iloc[row,:]["銘柄（コード）"].split()[0]
        name = df_stock_fund_nisa.iloc[row,:]["銘柄（コード）"].split()[1]
        symbols.append(symbol)
        names.append(name)
    df_stock_fund_nisa = format_data(df_stock_fund_nisa, '株式(現物/NISA預り)', '上場ＴＰＸ')

    # 結合
    df_result = pd.concat([df_stock_specific, df_stock_fund_nisa])
    df_result['code'] = symbols
    df_result["code"] = [df_result["code"].iloc[i]+".T" for i in range(len(df_result))]
    df_result['name'] = names
    df_result = df_result.drop(["ファンド名"],axis=1)
    df_result["評価額"] = [round(i) for i in df_result["評価額"]]
    df_result["損益（％）"] = [round(i) for i in df_result["損益（％）"]]

    return df_result

def get_table_en(driver):
    # 遷移するまで待つ
    time.sleep(0.5)

    # ポートフォリオの画面に遷移
    # driver.find_element(by=By.XPATH, value='//*[@id="link02M"]/ul/li[3]/a/img').click()
    driver.find_element(by=By.XPATH, value='//*[@id="navi02P"]/ul/li[2]/div/a').click()

    # 文字コードをUTF-8に変換
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupでパース
    soup = BeautifulSoup(html, "html.parser")
    symbols = []
    names = []
    curr_price = []
    profit_ratio = []
    categories = []

    # 米ドル/円の情報を含むtrタグを見つける
    usd_jpy_tr = None
    for tr in soup.find_all("tr", class_="mtext"):
        if "米ドル/円" in tr.text:
            usd_jpy_tr = tr
            break

    # 米ドル/円のレートを抽出する
    if usd_jpy_tr:
        usd_jpy_rate = float(usd_jpy_tr.find_all("td")[1].text)
        print("米ドル/円:", usd_jpy_rate)
    else:
        print("米ドル/円の情報が見つかりませんでした")
        
    # 米国株式（現物/NISA預り）
    table_data = soup.find_all("table", cellpadding="1", cellspacing="1", width="100%")
    df_stock_specific = pd.read_html(str(table_data), header=0)[0]
    df_stock_specific_A = df_stock_specific.iloc[::2,:]
    df_stock_specific_B = df_stock_specific.iloc[1::2,:]
    
    for row in range(len(df_stock_specific_A)):
        symbol = df_stock_specific_A.iloc[row,0].split()[0]
        name = df_stock_specific_A.iloc[row,0].split()[1]
        categories.append("現物/NISA預り")
        symbols.append(symbol)
        names.append(name)
        curr_price.append(
            float(df_stock_specific_B.iloc[row,2]) * float(df_stock_specific_B.iloc[row,0])\
                * usd_jpy_rate
            )
        profit_ratio.append(
            (
                (float(df_stock_specific_B.iloc[row,2])-float(df_stock_specific_B.iloc[row,1])) / \
                float(df_stock_specific_B.iloc[row,1])
            )*100.
        )
            
    df_en_result = pd.DataFrame()
    df_en_result["code"] = symbols
    df_en_result["name"] = names
    df_en_result["カテゴリー"] = categories
    df_en_result["評価額"] = [round(i) for i in curr_price]
    df_en_result["損益（％）"] = [round(i) for i in profit_ratio]
    
    return df_en_result

def get_balance(driver):
    # 遷移するまで待つ
    time.sleep(0.5)

    # ポートフォリオの画面に遷移
    driver.find_element(by=By.XPATH, value='//*[@id="link02M"]/ul/li[3]/a/img').click()

    # 文字コードをUTF-8に変換
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupでパース
    soup = BeautifulSoup(html, "html.parser")

    # 現金残高等（合計）の情報を含むdivタグを見つける
    cash_balance_div = None
    for div in soup.find_all("div", class_="margin"):
        font = div.find("font", class_="mtext")
        if font and "現金残高等（合計）" in font.get_text(strip=True):
            cash_balance_div = div
            break

    # 現金残高等（合計）の値を抽出する
    if cash_balance_div:
        cash_balance_value_td = cash_balance_div.find_next("td", class_="mtext")
        cash_balance = float(cash_balance_value_td.get_text(strip=True).replace(",", ""))
        print("現金残高等（合計）:", cash_balance)
    else:
        print("現金残高等（合計）の情報が見つかりませんでした")
        
    return cash_balance

def get_table(driver):
    df_ja_result = get_table_ja(driver)
    balance = get_balance(driver)
    df_en_result = get_table_en(driver)
    return pd.concat([df_ja_result, df_en_result], axis=0), balance