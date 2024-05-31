import requests
from bs4 import BeautifulSoup

# Function to extract text from URL
def extract_text_from_url(url):
    downloaded_page = requests.get(url)
    soup = BeautifulSoup(downloaded_page.content, "html.parser")
    paragraphs = soup.find_all('p', class_='dcr-jdlpgv')
    reportTextArr = []
    if paragraphs:
        for p in paragraphs:
            paragraph_text = p.get_text()
            reportTextArr.append(paragraph_text)
    return reportTextArr
