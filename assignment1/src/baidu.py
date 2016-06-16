#!/bin/env python2

#import requests
import multiprocessing
import concurrent.futures
import human_curl

FILE_NAME_IN = 'baidu_results.txt'
FILE_NAME_OUT = 'baidu_results_out.txt'
file = open(FILE_NAME_IN, 'r')

def main(line):
    try:
        out = open(FILE_NAME_OUT, 'a')
        strLine = str(line)
        [id, link] = strLine.split('link = ')
        url = link.rstrip("\n")
        e = human_curl.get(url)
        head = e.headers['location']    # http request on header for location which contains the url
        out.write(id + "link = " + head + "\n")          # output to file
        #out.flush()
        print head
    except:
        print "bad url: " + url

pool = concurrent.futures.ProcessPoolExecutor(2)       # start x worker processes
future = [pool.submit(main, line)
    for line in file if not line.strip() == ""]
concurrent.futures.wait(future)