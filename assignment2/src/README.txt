Runs in Python 3.4.3

parseData.py is the only file. Expects data files 
  movies.dat  
  testFile.csv
  trainingRatings.dat
to be placed in a local folder named data

To run:
  python parseData.py method [dataset]
where method is the method to use and dataset can be test (default) or training, 
to run on the test set and training set respectively.
Will export results in file
  submissionFile[method].csv if test
  trainingResultFile[method].csv if training

Methods:
  simple1       - average rating over all movies
  simple2       - average rating by user
  simple3       - average rating of movie overall users
  content
  collaborative - very slow, avg. entry processed: 1/sec
  hybrid        - expects content and collaborative to already have been runned.

Note running time can be long (is very long on collaborative) and memory usage 
is large, usually more than 3 gb RAM. Threaded versions are implemented, but 
these are for some versions slower, as Python copies memory for each thread. 

Examples:
Run the simple3 method on the test (validation) set. Outputs results in 
submissionFileSimple3.csv:
  python parseData.py simple3 test

Run the content-based method on the training data and print RMSE to terminal 
(stdout). Outputs trainingResultFileContentBased.csv:
  python parseData.py content training