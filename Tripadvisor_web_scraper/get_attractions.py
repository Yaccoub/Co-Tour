from bs4 import BeautifulSoup
import csv
from datetime import datetime
import re
from selenium import webdriver
from selenium.common import exceptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException


def main():
    global fileName
    fileName = "Eisbach Wave.csv"
    global titleList
    titleList = []
    global writer
    fw = open(fileName, "w", newline='', encoding="utf-8")
    writer = csv.writer(fw, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['date', 'title', 'text', 'rating', 'visitor_origin', 'visit'])
    url = "https://www.tripadvisor.com/Attraction_Review-g187309-d4609080-Reviews-Eisbach_Wave-Munich_Upper_Bavaria_Bavaria.html"
    options = webdriver.ChromeOptions()
    options.add_argument('--lang=en')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    ignored_exceptions = (NoSuchElementException, StaleElementReferenceException)
    driver.implicitly_wait(6)
    all_languages = WebDriverWait(driver, 60, ignored_exceptions=ignored_exceptions).until(find_languages)
    all_languages.click()
    try:
        driver.implicitly_wait(4)
        button = WebDriverWait(driver, 40, ignored_exceptions=ignored_exceptions).until(findReadmore)
        button.click()
    except exceptions.StaleElementReferenceException as e:
        print(e)
        pass



    iteration = 0
    totalNumPages = 50
    analyzeIndexPage(driver)
    while url != None and iteration < totalNumPages:
        iteration = iteration + 1
        driver.implicitly_wait(4)
        for i in range(4):
            try:
                driver.implicitly_wait(4)
                Next = WebDriverWait(driver, 40, ignored_exceptions=ignored_exceptions).until(findNext)
                Next.click()

                break
            except exceptions.StaleElementReferenceException as e:
                print(e)
                pass


        for i in range(4):
            try:
                driver.implicitly_wait(4)
                button = WebDriverWait(driver, 40, ignored_exceptions=ignored_exceptions).until(findReadmore)
                button.click()
                break
            except exceptions.StaleElementReferenceException as e:
                print(e)
                pass
            try:
                driver.implicitly_wait(6)
                all_languages = WebDriverWait(driver, 60, ignored_exceptions=ignored_exceptions).until(find_languages)
                all_languages.click()
                break
            except exceptions.StaleElementReferenceException as e:
                print(e)
                pass



        analyzeIndexPage(driver)
        print("iter %s finished..." % (str(iteration)))


def findReadmore(driver):
    element = driver.find_elements_by_xpath(
        "//span[starts-with(@class,'_3maEfNCR')]")
    if element:
        print('found button readmore')
        return element[0]
    else:
        return False

def find_languages(driver):
    element = driver.find_elements_by_xpath("//span[starts-with(@class,'_1wk-I7LS')]")[0]
    if element:
        print('found button All Languages')
        return element
    else:
        return False

def findNext(driver):
    element = driver.find_elements_by_xpath("//a[@class='ui_button nav next primary ']")[0]
    if element:
        print('found button Next')
        return element
    else:
        return False

def analyzeIndexPage(driver):
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    listReviews = []
    listTitles = []
    listLocations = []
    listRatings = []
    listExpDates = []
    listVisittype = []
    for section in soup.find_all("div", attrs={"class": re.compile(r"Dq9MAugU T870kzTX LnVzGwUB")}):
        review = section.find("q", attrs={
            "class": re.compile(r"IRsGHoPm")})
        content = review.findChildren("span")[0].get_text()
        if content != None:
            listReviews.append(content)
        else:
            listReviews.append("")

        title = section.find("a", attrs={
            "class": re.compile(r"ocfR3SKN")})

        text = title.findChildren("span")[0].get_text()
        if text != None:
            listTitles.append(text)
        else:
            listTitles.append("")

        rate = section.find("span", attrs={"class": re.compile(r"^ui_bubble_rating bubble_.*")})
        if rate != None:
            listRatings.append(rate["class"][1][7:9])
        else:
            listRatings.append("")

        visit = section.find("span", attrs={
            "class": re.compile(r"_2bVY3aT5")})
        if visit != None:
            listVisittype.append(visit.get_text())
        else:
            listVisittype.append("")

        location = section.find("span",
                                attrs={"class": re.compile(r"default _3J15flPT small")})
        if location != None:
            listLocations.append(location.get_text())
        else:
            listLocations.append("")

        expdate = section.find("span", attrs={
            "class": re.compile(r"34Xs-BQm")})
        if expdate != None:
            listExpDates.append(expdate.get_text())
        else:
            listExpDates.append("")
    for i in range(0, 5):
        writer.writerow((listExpDates[i], listTitles[i], listReviews[i], listRatings[i], listLocations[i], listVisittype[i]))
    return None


if __name__ == '__main__':
    main()