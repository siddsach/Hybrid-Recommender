import re
import csv
import requests
from bs4 import BeautifulSoup

START = 'http://www.springfieldspringfield.co.uk/movie_scripts.php?order=0'


def get_letter_urls(start_url):
    '''
    Given a start url, find all the 27 urls (A-Z and 0) that are the movie
    lists for each letter.
    '''
    letter_urls = []

    result = requests.get(start_url)
    c = result.content
    soup = BeautifulSoup(c, 'lxml')
    saa = soup.find_all(class_='search-and-alpha')[0]
    links = saa.find_all('a')
    for link in links:
        letter_urls.append('http://www.springfieldspringfield.co.uk' + link['href'])

    return letter_urls


def get_num_pages(letter_url):
    '''
    Given a letter url, return how many pages there are for that letter.
    '''
    result = requests.get(letter_url)
    c = result.content
    soup = BeautifulSoup(c, 'lxml')
    mcl0 = soup.find_all(class_='main-content-left')[0]
    num_pages = int(mcl0.find_all('a')[-1].text)
    return num_pages


def get_script_urls(letter_url):
    '''
    Given a base url for each letter, creates a list of links (strings) to
    all movie scripts starting with that letter.
    '''
    urls_for_letter = []
    num_pages = get_num_pages(letter_url)

    for i in range(num_pages):
        p = i + 1 # Page 1 is first on the website
        print('Gathering script URLs from page ' + str(p) + '...')
        current_url = letter_url + '&page=' + str(p)
        print('The URL is ' + current_url)
        result = requests.get(current_url)
        c = result.content
        
        soup = BeautifulSoup(c, 'lxml')
        mcl = soup.find_all(class_='main-content-left')[0]
        links = mcl.find_all('a', class_='script-list-item')
        
        for link in links:
            urls_for_letter.append('http://www.springfieldspringfield.co.uk'\
                + link['href'])
            print(link.text)

    return urls_for_letter


def get_script(script_url):
    '''
    Returns a string with the movie script, when given the script's URL.
    '''
    result = requests.get(script_url)
    c = result.content
    soup = BeautifulSoup(c, 'lxml')
    try:
        ssc = soup.find_all(class_='scrolling-script-container')[0]
    except IndexError:
        return ('', '')
    title = soup.find_all('h1')[0]
    return ssc.text, title.text[:-13]


def go():
    all_script_urls = []

    print('Getting URLs for each letter...')
    letter_urls = get_letter_urls(START)
    print('Letter URLs retrieved.')

    for letter_url in letter_urls:
        print('Now getting script URLs for letter ' + letter_url[-1] + '...')
        letter_script_urls = get_script_urls(letter_url)
        all_script_urls.extend(letter_script_urls)
        print('All script URLs for letter ' + letter_url[-1]
            + ' have been scraped.')

    print('Now going through list of script URLs and scraping them...')
    with open('scripts.csv', 'wt') as result_file:
        wr = csv.writer(result_file)
        wr.writerow(['title', 'script'])
        for script_url in all_script_urls:
            script, title = get_script(script_url)
            if not script == '':
                row = [title, script.strip()]
                wr.writerow(row)
            print(title + ' scraped')


go()