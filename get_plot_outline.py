
from bs4 import BeautifulSoup
import requests
import re

def retrieve_plot(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")

    # extract the sequence of sections
    # this is needed to stitch the regex together to exactly locate the plot description
    # starting of Plot: <span class="mw-headline" id="Plot">Plot</span> ... </h2>\n<p>
    # closing of Plot: </p>\n<h2><span class="mw-headline" id="Cast">Cast</span>
    sections = [tag.get_text() for tag in soup.find_all("span", class_="mw-headline")]
    list_index_plot = sections.index("Plot")

    content = soup.getText().splitlines()
    regex_res = re.findall('Plot(.+?)' + sections[list_index_plot + 1] + '', str(content))
    plot = regex_res.pop(1)
    return plot


urls = ["https://en.wikipedia.org/wiki/Iron_Man_(2008_film)", "https://en.wikipedia.org/wiki/The_Incredible_Hulk_(film)",
        "https://en.wikipedia.org/wiki/Iron_Man_2", "https://en.wikipedia.org/wiki/Thor_(film)", "https://en.wikipedia.org/wiki/Captain_America:_The_First_Avenger",
        "https://en.wikipedia.org/wiki/The_Avengers_(2012_film)", "https://en.wikipedia.org/wiki/Iron_Man_3",
        "https://en.wikipedia.org/wiki/Thor:_The_Dark_World", "https://en.wikipedia.org/wiki/Captain_America:_The_Winter_Soldier",
        "https://en.wikipedia.org/wiki/Guardians_of_the_Galaxy_(film)", "https://en.wikipedia.org/wiki/Avengers:_Age_of_Ultron",
        "https://en.wikipedia.org/wiki/Ant-Man_(film)", "https://en.wikipedia.org/wiki/Captain_America:_Civil_War",
        "https://en.wikipedia.org/wiki/Doctor_Strange_(2016_film)", "https://en.wikipedia.org/wiki/Guardians_of_the_Galaxy_Vol._2",
        "https://en.wikipedia.org/wiki/Spider-Man:_Homecoming", "https://en.wikipedia.org/wiki/Thor:_Ragnarok",
        "https://en.wikipedia.org/wiki/Black_Panther_(film)", "https://en.wikipedia.org/wiki/Avengers:_Infinity_War",
        "https://en.wikipedia.org/wiki/Ant-Man_and_the_Wasp", "https://en.wikipedia.org/wiki/Captain_Marvel_(film)",
        "https://en.wikipedia.org/wiki/Avengers:_Endgame", "https://en.wikipedia.org/wiki/Spider-Man:_Far_From_Home"]


plot_dict = {}

for url in urls:
    plot = retrieve_plot(url)
    plot_dict[url] = plot