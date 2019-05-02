from bs4 import BeautifulSoup as bs
import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd
import re

baseurl = "http://www.espn.com/mens-college-basketball/statistics/player/_/stat/"
max_rank_tracked = 99
DATA_DIR = "./data/"

class ESPNScraper():
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        self.driver = webdriver.Chrome(chrome_options=chrome_options, executable_path="./lib/chromedriver")
        self.soup = None

    # Gets links to various stats on a per-year basis. Run once per year/season.
    def getAllStatLinks(self):
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

        def buildTail(n):
            return "/count/" + str(n+1)

        def pickle_all(dflist):
            print("Pickling", len(dflist), "dataframes")
            for i, df in enumerate(dflist):
                df.to_pickle(DATA_DIR + 'df_' + df.meta)
            print(len(dflist), "dataframes pickled!")

        self.reload(baseurl, resoup=True)
        years = self.getDropdownLinks()
        dfs = []
        for year_link in years:
            curr_year = re.findall(r'[0-9]{4}', year_link)
            curr_year = "2019" if curr_year == [] else curr_year[0]
            print("Fetching data for year", curr_year)
            self.reload(year_link, resoup=True)
            stat_links = self.getAllStatLinks()
            df = None
            for i, stat_page in enumerate(stat_links): # ~1 min. per iteration
                print("Fetching stat ({0}/{1}) from {2}...".format(i + 1, len(stat_links), stat_page))
                players_read = 0
                tail = ""
                header = None
                postseason_flag = None
                while players_read < max_rank_tracked:
                    self.reload(stat_page + tail, resoup=True)
                    all_entries = self.soup.findAll("tr")
                    table_entries = [entry for entry in all_entries if entry["class"][0] != "colhead"]
                    if header is None: 
                        header = [title.text for title in self.soup.find_all("tr", {"class": "colhead", "align": "right"})[0]] + ['YR', 'POST']
                        df = pd.DataFrame(columns=header)
                        # print(header)
                    if postseason_flag is None: postseason_flag = 0 if "seasontype/2" in stat_page else 1
                    last_rank = 0
                    for j, entry in enumerate(table_entries):
                        row = [d.text for d in entry.findAll("td")]
                        try:
                            rank = int(row[0])
                            last_rank = j + players_read + 1
                        except ValueError:
                            row[0] = last_rank # tied
                        row += [curr_year, postseason_flag]
                        df.loc[len(df),:] = row
                        # print(row)
                    players_read += len(table_entries) # batch update at the end so tied rank behavior works properly
                    tail = buildTail(players_read)
                # print(df.head(n=20))
                df.meta = '_'.join([curr_year, str(i)])
                dfs.append(df)
        pickle_all(dfs)

    def reload(self, url, resoup=False):
        if not url.startswith("http:"): url = "http:" + url
        self.driver.get(url)
        if resoup: self.soup = bs(self.driver.page_source, features="html5lib") # default parser is lxml, which works poorly


if __name__ == '__main__':
    print("Creating scraper...")
    scraper = ESPNScraper()
    scraper.scrape()
