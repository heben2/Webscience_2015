
The solution uses Python 3.4 and 2.7 and sqlite3. It is functional on Linux 
(no other platform tested). It does however not produce the expected results 
(from the GoogleScraper part), as described in the report.

Stand in folder src/
Create python environment and activate it
    virtualenv -p /usr/bin/python env
    source env/bin/activate
Now install requirements:
    pip install -r requirements1.txt
You should now be able to run the code.
Run scraper
    GoogleScraper --keyword-file queries/queries_all.txt -s "baidu,bing,yandex" --num-pages-for-keyword 20 -m selenium

The results are placed in the file google_scraper.db
After GoogleScraper, extract all baidu links from the database to file 
baidu_results.txt using sqlite3:
    sqlite3
    .open google_scraper.db
    .mode line
    .output baidu_results.txt
    select 'id = ' || a.serp_id || ' link = ' || a.link from link a, serp b where b.search_engine_name = 'baidu' and a.serp_id = b.id;

Exit sqlite3.

Stand in src/
Create python environment and activate it
    virtualenv -p /usr/bin/python2 venv
    source venv/bin/activate
Now install requirements:
    pip install -r requirements2.txt
You should now be able to run the code.
Run the baidu link converter
    python baidu.py 
This should create the file baidu_results_out.txt.
Exit environment:
    deactivate
The result is not in csv-format (because I did not know it needed to be at 
first), so run 
    python convert_baidu_results_to_csv.py
to convert it to csv-format in the file baidu_results_out.csv.

Now import the results in the database
    sqlite3
    .open google_scraper.db
    .mode line
    create table baidu_links(id integer, link varchar);
    .separator ","
    .import baidu_results_out.csv baidu_links

It is now possible to do math on these results, e.g.

-- get shared queries
select distinct(a.query) from serp a, serp b where a.search_engine_name = 'bing' and b.search_engine_name = 'baidu' and a.query = b.query

-- sum total number of search results on shared queries
select sum(num_results) from serp a, (select distinct(a.query) from serp a, serp b where a.search_engine_name = 'bing' and b.search_engine_name = 'baidu' and a.query = b.query) b where a.query = b.query and a.search_engine_name = 'bing';

select sum(num_results) from serp a, (select distinct(a.query) from serp a, serp b where a.search_engine_name = 'bing' and b.search_engine_name = 'baidu' and a.query = b.query) b where a.query = b.query and a.search_engine_name = 'baidu';

-- Count number of retrieved unique links
select count(distinct(link)) from link a, serp b, (select distinct(a.query) from serp a, serp b where a.search_engine_name = 'bing' and b.search_engine_name = 'baidu' and a.query = b.query) c where a.serp_id = b.id and b.query = c.query and b.search_engine_name = 'bing';

select count(distinct(d.link)) from link a, serp b, (select distinct(a.query) from serp a, serp b where a.search_engine_name = 'bing' and b.search_engine_name = 'baidu' and a.query = b.query) c, baidu_links d where a.serp_id = b.id and b.query = c.query and a.id = d.id and b.search_engine_name = 'baidu';
