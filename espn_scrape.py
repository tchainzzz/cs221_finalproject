from bs4 import BeautifulSoup as bs
import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd

baseurl = "http://www.espn.com/mens-college-basketball/statistics/player/_/stat/"

class ESPNScraper():
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        self.driver = webdriver.Chrome(chrome_options=chrome_options, executable_path="./chromedriver")
        self.soup = None

    # Gets links to various stats on a per-year basis. Run once per year/season.
    def getAllLinks(self):
        all_links = self.soup.findAll("a", href=True)
        raw_links = []
        for link in all_links:
            if link.has_attr("href") and link["href"].startswith(baseurl[len("http:"):]) and "sort" not in link["href"]:
                raw_links.append(link["href"])
        return raw_links

    # Gets links to each year. Run once.
    def getDropdownLinks(self):
        drop_down = self.soup.findAll("option")
        drop_down_values = [d["value"] for d in drop_down]
        return drop_down_values

    def scrape(self):
        self.driver.get(baseurl)
        self.soup = bs(self.driver.page_source, features="html5lib") # default parser is lxml, which works poorly
        links = self.getAllLinks()
        years = self.getDropdownLinks()


if __name__ == '__main__':
    print("Creating scraper...")
    scraper = ESPNScraper()
    scraper.scrape()
