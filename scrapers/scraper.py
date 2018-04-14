import logging
import json
from grab.spider import Spider, Task
import os
import urllib
import warnings
import telegram_send
from datetime import datetime
import csv
PATH_TO_DATA = '/media/grigory/Data/WIKIDATA' # '/media/grigory/Диск/WIKIDATA'



class WikiSource(Spider):
    #initial_urls = [r'https://ru.wikisource.org/wiki/Категория:Русская_литература',] #r'https://ru.wikisource.org/wiki/Категория:Литература']

    def prepare(self):
        self.base_url = r'https://ru.wikisource.org'
        self.articles_counter = 0
        self.corrupted_articles_counter = 0
        self.corrupted_categories_counter = 0
        self.parsed_articles = set()
        self.create_grab_instance(time)
        start_categories = [r'Русская_драматургия‎',
                            r'Русская_поэзия',
                            r'Русская_проза‎',
                            r'Русский_футуризм‎',
                            r'Русский_символизм‎']
        self.initial_urls = [r'https://ru.wikisource.org/wiki/Категория:'+ name for name in start_categories]

        self.csv_fields = ('id', 'categories', 'article_url', 'name', 'author', 'author_url', 'path')
        with open('log.csv', 'w') as f:
            writer = csv.DictWriter(f, delimiter=',', quotechar='/', fieldnames=self.csv_fields)
            writer.writeheader()

        try:
            os.mkdir(PATH_TO_DATA)
        except FileExistsError:
            pass

    def task_initial(self, grab, task):
        yield Task('handle_category', url=task.url, art_path=[task.url[len(self.base_url):]])

    def task_handle_category(self, grab, task):
        try:
            header = grab.doc.select('//h1[@id="firstHeading"]').text()[len(r'Категория:'):]
        except IndexError:
            warnings.warn('Invalid category: "{}". Ignoring...'.format(task.url))
            self.corrupted_categories_counter += 1
            return

        subcategories = grab.doc.select(r'//div[@class="CategoryTreeItem"]')
        for subcat in subcategories:
            href = subcat.select(r'a//@href').text()
            yield Task('handle_category', url=self.base_url+href, art_path=task.art_path + [header])

        articles = grab.doc.select(r'//div[@id="mw-pages"]//li')
        for article in articles:
            href = article.select(r'a//@href').text()
            yield Task('handle_article', url=self.base_url+href, art_path=task.art_path + [header])

    def task_handle_article(self, grab, task):
        if grab.doc.select(r'//span[@id="ws-name"]'):
            # Значит эта страница  - страница с  автором, а не с произведением.
            yield Task('handle_author', url=task.url, art_path=task.art_path)
        else:
            try:
                article_url = urllib.parse.unquote(task.url[len(self.base_url):])
                # Проверяем, что мы ещё не работали  с  этой статьи
                if article_url in self.parsed_articles:
                    warnings.warn('Duplicate article entry.')
                    return

                categories = grab.doc.select(r'//div[@id="mw-normal-catlinks"]//ul//li')
                categories_data = [cat.text() for cat in categories]
                name = grab.doc.select(r'//span[@id="ws-title"]').text()
                author = grab.doc.select(r'//span[@id="ws-author"]//a//@title').text()
                author_url = grab.doc.select(r'//span[@id="ws-author"]//a//@href').text()
                article = {
                    'categories': categories_data,
                    'body': grab.doc.body.decode('utf-8'),
                    'article_url': article_url,
                    'name': name,
                    'author': author,
                    'author_url': urllib.parse.unquote(self.base_url+author_url),
                    'path': task.art_path
                }

                # Записываем полную информацию о статье как отдельный json файл.
                with open(os.path.join(PATH_TO_DATA, str(self.articles_counter)+'.json'), 'w') as f:
                    json.dump(article, f, indent=4, ensure_ascii=False)

                # Дописываем краткую инфу о статье в log файл.
                with open('log.csv', 'a') as f:
                    writer = csv.DictWriter(f, delimiter=',', quotechar='/', fieldnames=self.csv_fields)
                    short_info= {
                        'id': self.articles_counter,
                        'categories': '##'.join(categories_data),
                        'article_url': article_url,
                        'name': name,
                        'author': author,
                        'author_url': urllib.parse.unquote(author_url),
                        'path': '##'.join(task.art_path)
                    }
                    writer.writerow(short_info)
                self.articles_counter += 1
                self.parsed_articles.add(article_url)
                print(self.articles_counter)
                if self.articles_counter % 25000 == 0:
                    self.send_telegramlog()

            except IndexError as e:
                warnings.warn(r'Error parsing article: "{}"'.format(task.url))
                warnings.warn(str(task.art_path))
                warnings.warn(e)
                self.corrupted_articles_counter += 1

    def task_handle_author(self, grab, task):
        try:
            articles_selector = r'//div[@class="mw-parser-output"]//li//a[not(contains(@class,"external text"))]'
            author_articles = grab.doc.select(articles_selector )
            for article in author_articles:
                try:
                    href = article.select(r'@href').text()
                    new_url = self.base_url +  (r'/wiki/'  if href[:len(r'/wiki/')] != r'/wiki/' else '') + href
                    yield Task('handle_article', url=new_url, art_path=task.art_path)
                except IndexError:
                    warnings.warn('Invalid article "{}" from author page: {}'.format(article.text(), task.url))
        except NameError:
            pass

    def send_telegramlog(self):
        articles_mess = '{:%D, %H:%M}:: Articles ready: {}, Ignored corrupted articles: {}'.format(datetime.now(),
                                                                                  self.articles_counter,
                                                                                  self.corrupted_articles_counter)
        files_ = os.path.getsize(os.path.join(PATH_TO_DATA, f) for f in os.listdir(PATH_TO_DATA) \
                                                    if os.path.isfile(os.path.join(PATH_TO_DATA,f)))
        total_size = sum(files_) / (1000 ** 3)
        size_mess = 'Total size of data dir: {}GB'.format(total_size)
        try:
            telegram_send.send([articles_mess, size_mess])
        except:
            pass


if __name__ == '__main__':
    start = datetime.now()
    print(start.strftime('%y.%m.%d - %H:%M'))
    #logging.basicConfig(level=logging.DEBUG)
    spider = WikiSource(thread_number=3)
    #spider.setup_queue(backend='mongo', database='queue_db')  # очередь в монго
    spider.run()