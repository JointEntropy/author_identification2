import logging
import json
import pickle
from grab.spider import Spider, Task
import os
import urllib
import warnings
import telegram_send
from datetime import datetime
import csv
PATH_TO_DATA = '/media/grigory/Data/WIKIDATA' # '/media/grigory/Диск/WIKIDATA'
from functools import reduce


class WikiSource(Spider):
    # initial_urls = [r'https://ru.wikisource.org/wiki/Категория:Русская_литература',] #r'https://ru.wikisource.org/wiki/Категория:Литература']
    def create_grab_instance(self, **kwargs):
        g = super(WikiSource, self).create_grab_instance(**kwargs)
        g.setup(timeout=200, connect_timeout=200)
        return g

    def prepare(self):
        self.base_url = r'https://ru.wikisource.org'
        with open('bad_guys.pckl', 'rb') as fr:
            self.bad_guys = pickle.load(fr)
        with open('bad_guys_shortcuts.pckl', 'rb') as fr:
            self.bad_guys_sc = pickle.load(fr)
        self.articles_counter = 0
        self.corrupted_articles_counter = 0
        self.corrupted_categories_counter = 0
        self.parsed_articles = set()

        self.initial_urls = [r'https://ru.wikisource.org/wiki/Служебная:Все_страницы']

        self.csv_fields = ('id', 'categories', 'article_url', 'name', 'author',
                           'translator', 'author_url', 'verified',
                           'content_len',
                           'oldspell')
        with open('log.csv', 'w') as f:
            writer = csv.DictWriter(f, delimiter=',', quotechar='/', fieldnames=self.csv_fields)
            writer.writeheader()
        try:
            os.mkdir(PATH_TO_DATA)
        except FileExistsError:
            pass

    def task_initial(self, grab, task):
        yield Task('handle_searchpage', url=task.url)

    def task_handle_searchpage(self, grab, task):
        articles = grab.doc.select(r'//ul[@class="mw-allpages-chunk"]//li')
        for article in articles:
            href = article.select(r'a//@href').text()
            # Проверяем, что это не статья из какой-нибудь энциклопедии
            href_c = urllib.parse.unquote(href)
            if not any(bad_author in href_c for bad_author in self.bad_guys_sc):
                yield Task('handle_article', url=self.base_url+href)
            else:
                print('Bad author in {}'.format(href_c))
        next_page_url = grab.doc.select(r'//div[@class="mw-allpages-nav"]//a//@href')[-1].text()
        print(urllib.parse.unquote(next_page_url))
        yield Task('handle_searchpage', url=self.base_url+next_page_url)#, priority=0)

    def task_handle_article(self, grab, task):
        article_url = urllib.parse.unquote(task.url[len(self.base_url):])
        # Проверяем, что мы ещё не работали  с  этой статьи
        if article_url in self.parsed_articles:
            #warnings.warn('Duplicate article entry.')
            return

        author = grab.doc.select(r'//span[@id="ws-author"]//a//@title')
        if author:
            author = author.text()
            author_url = grab.doc.select(r'//span[@id="ws-author"]//a//@href').text()
            if author in self.bad_guys:
                warnings.warn('Just another dict page: "{}". Ignore..'.format(urllib.parse.unquote(task.url)))
                return
        else:
            warnings.warn('No author provided. Ignore...')
            return

        categories = grab.doc.select(r'//div[@id="mw-normal-catlinks"]//ul//li')
        categories_data = [cat.text() for cat in categories]

        try:
            name = grab.doc.select(r'//span[@id="ws-title"]').text()
            translator = grab.doc.select(r'//span[@id="ws-translator"]//a//@title')
            verified = False if len(grab.doc.select(r'//div[@id="mw-fr-revisiontag"]')) == 1 else True
            old_spell = bool(len(grab.doc.select(r'//div[@class="oldspell"]')))
            # TODO тут нужно всё таки чекать какие именно абзацы мы будем записывать
            content = '\n\n'.join(p.text() \
                             for p in grab.doc.select(r'//div[@class="mw-parser-output"]//p'))
            #                  if p.getparent().get('class') is not None)
            # Записываем полную информацию о статье как отдельный json файл.
            article = {
                'categories': categories_data,
                'body': grab.doc.body.decode('utf-8'),
                'article_url': article_url,
                'name': name,
                'author': author,
                'translator': translator.text() if len(translator) else None,
                'author_url': urllib.parse.unquote(self.base_url+author_url),
                'verified': verified,
                'content': content,
                'oldspell': old_spell
            }
            with open(os.path.join(PATH_TO_DATA, str(self.articles_counter)+'.json'), 'w') as f:
                json.dump(article, f, indent=4, ensure_ascii=False)

            # Дописываем краткую инфу о статье в log файл.
            with open('log.csv', 'a') as f:
                writer = csv.DictWriter(f, delimiter=',', quotechar='/', fieldnames=self.csv_fields)
                short_info = {
                    'id': self.articles_counter,
                    'categories': '##'.join(categories_data),
                    'article_url': article_url,
                    'name': name,
                    'author': author,
                    'translator': translator.text() if len(translator) else None,
                    'author_url': urllib.parse.unquote(author_url),
                    'verified': verified,
                    'content_len': len(content),
                    'oldspell': old_spell
                }
                writer.writerow(short_info)
            self.articles_counter += 1
            self.parsed_articles.add(article_url)
            print(self.articles_counter)
            if self.articles_counter % 20000 == 0:
                self.send_telegramlog()

        except IndexError as e:
            warnings.warn(r'Error parsing article: "{}"'.format(task.url))
            self.corrupted_articles_counter += 1

    def send_telegramlog(self):
        articles_mess = '{:%D, %H:%M}:: Articles ready: {}'.format(datetime.now(), self.articles_counter)
        # files_ = [os.path.getsize(os.path.join(PATH_TO_DATA, f)) for f in os.listdir(PATH_TO_DATA) \
        #                                             if os.path.isfile(os.path.join(PATH_TO_DATA,f))]
        #total_size = sum(files_) / (1000 ** 3)
        #size_mess = 'Total size of data dir: {:.3f}GB'.format(total_size)
        try:
            telegram_send.send([articles_mess,]) #size_mess])
        except:
            pass


if __name__ == '__main__':
    start = datetime.now()
    print(start.strftime('%y.%m.%d - %H:%M'))
    spider = WikiSource(thread_number=2)
    spider.run()