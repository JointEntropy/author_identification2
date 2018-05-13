"""
Scraper для
https://loveread-ec.appspot.com

# TODO
...петролекс.
"""
import logging
import json
from grab.spider import Spider, Task
import os
import urllib
import warnings
import telegram_send
from datetime import datetime
import csv
PATH_TO_DATA = '/media/grigory/Data/LoveReadDetective' # '/media/grigory/Диск/WIKIDATA'
logname = '../data/loveread_detective_log.csv'
genre = 3
max_pages = 1645
from time import sleep

class Loveread(Spider):
    def create_grab_instance(self, **kwargs):

        g = super(Loveread, self).create_grab_instance(**kwargs)
        # g.setup(proxy='151.106.29.120:1080', proxy_type='http')
        # g.load_proxylist('proxies.txt', source_type='text_file')
        return g

    def prepare(self):

        self.base_url = r'https://loveread-ec.appspot.com'
        self.read_href = r'https://loveread-ec.appspot.com/read_book.php?id={id}&p={p}'
        self.articles_counter = 0
        self.create_grab_instance()
        self.initial_urls = [r'https://loveread-ec.appspot.com/index_book.php?id_genre={genre}&p={page}'\
                                                        .format(genre=genre, page=i) for i in range(max_pages)]

        self.csv_fields = ('id',  'name', 'author', 'url', 'translate', 'page')
        with open(logname, 'w') as f:
            writer = csv.DictWriter(f, delimiter=',', quotechar='/', fieldnames=self.csv_fields)
            writer.writeheader()

        try:
            os.mkdir(PATH_TO_DATA)
        except FileExistsError:
            pass

    def task_initial(self, grab, task):
        yield Task('handle_category', url=task.url, art_path=[task.url[len(self.base_url):]])

    def task_handle_category(self, grab, task):
        articles = grab.doc.select(r'//a[@title="читать книгу"]//@href')
        descs = grab.doc.select('//td[@class="span_str"]')
        for article, desc in zip(articles, descs):
            href = article.text()
            translate = any([s.text() == 'Перевод книги:' for s in desc.select('//span')])
            read_id = href.rsplit('=')[-1]
            read_href = self.read_href.format(id=read_id, p=1)
            # if not translate:
            yield Task('handle_article', read_id=read_id, url=read_href, translate=translate, page=1)

    def task_handle_article(self, grab, task):
        title, author = list(grab.doc.select('//td[@class="tb_read_book"]//h2/a'))

        # article_url = urllib.parse.unquote(task.url[len(self.base_url):])
        pars = grab.doc.select('//p[@class="MsoNormal"]')
        text = []
        for p in pars:
            text.append(p.text())
        article = {
            'content': '\n\n'.join(text),
            'name': title.text(),
            'author': author.text()
        }
        with open(os.path.join(PATH_TO_DATA, str(self.articles_counter)+'.json'), 'w') as f:
            json.dump(article, f, indent=4, ensure_ascii=False)
        with open(logname, 'a') as f:
            writer = csv.DictWriter(f, delimiter=',', quotechar='/', fieldnames=self.csv_fields)
            short_info= {
                'id': self.articles_counter,
                'translate': task.translate,
                'name': title.text(),
                'author': author.text(),
                'url': urllib.parse.unquote(task.url[len(self.base_url):]),
                'page': task.page
            }
            writer.writerow(short_info)
        self.articles_counter += 1

        if task.page == 1:
            last_page = int(grab.doc.select('//div[@class="navigation"]//a')[-2].text())
            for p in range(2, last_page):
                read_href = self.read_href.format(id=task.read_id, p=p)
                yield Task('handle_article', read_id=task.read_id, url=read_href, translate=task.translate, page=p)
        print('{}:: {}'.format(self.articles_counter, urllib.parse.unquote(task.url[len(self.base_url):])))


if __name__ == '__main__':

    start = datetime.now()
    print(start.strftime('%y.%m.%d - %H:%M'))
    spider = Loveread(thread_number=1)
    #spider.setup_queue(backend='mongo', database='queue_db')  # очередь в монго
    spider.run()