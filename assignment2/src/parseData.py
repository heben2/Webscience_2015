#!/bin/env python3

from multiprocessing import Queue, Process, Pipe
from operator import itemgetter
import concurrent.futures
import sys
import warnings
import math


FILE_PATH_MOVIES = "data/movies.dat"
FILE_PATH_TAGS = "data/tags.dat"
FILE_PATH_TRAININGRATINGS = "data/trainingRatings.dat"

FILE_PATH_TEST = "data/testFile.csv"

# Used by simple1 as a constant once set
__average_rating = -1

__data_movies_loaded = False
MOVIES_GENRES = {} #movie_id -> [genre]
GENRES_MOVIES = {} #genre -> [movie_id]

__data_training_loaded = False
USERID_MOVIE_RATING = {} #user_id -> {movie -> rating}
MOVIE_USERID = {} #movie_id -> [user_id]

__data_movie_rating_loaded = False
MOVIE_RATING = {} #movie_id -> [rating]





# LOADING/INPUT METHODS

# Creates a .csv-file with given file_name.
# Truncates file if already existing
# Returns file handler to file with correct header, ready for use
def init_submission_file(file_name):
    out = open(file_name+".csv", 'w+')
    head = "Id,Rating"
    out.write(head + "\n")
    return out

def load_training_file():
    return open(FILE_PATH_TRAININGRATINGS, 'r')

def load_test_file():
    return open(FILE_PATH_TEST, 'r')

def load_movies():
    global MOVIES_GENRES, GENRES_MOVIES, __data_movies_loaded
    for line in open(FILE_PATH_MOVIES):
        fields = line.strip().split("::")
        genres = parse_genres(fields[2])
        movie_id = int(fields[0])
        MOVIES_GENRES[movie_id] = genres
        for genre in genres:
            if genre in GENRES_MOVIES:
                GENRES_MOVIES[genre].append(movie_id)
            else:
                GENRES_MOVIES[genre] = [movie_id]
    __data_movies_loaded = True

def load_trainingRating():
    global USERID_MOVIE_RATING, MOVIE_USERID, __data_training_loaded
    count = 0
    #userID::movieID::Rating::timestamp
    for line in open(FILE_PATH_TRAININGRATINGS):
        fields = line.strip().split("::")
        user_id = int(fields[0])
        movie_id = int(fields[1])
        rating = float(fields[2])
       
        #Debugging
        if count % 1000000 == 0:
            print(count)
        count += 1
        ##########

        if user_id in USERID_MOVIE_RATING:
            USERID_MOVIE_RATING[user_id][movie_id] = rating
            #USERID_MOVIE_RATING[user_id].append((movie_id, rating))
        else:
            USERID_MOVIE_RATING[user_id] = {movie_id: rating}

        if movie_id in MOVIE_USERID:
            MOVIE_USERID[movie_id].append(user_id)
        else:
            MOVIE_USERID[movie_id] = [user_id]
    __data_training_loaded = True


# For testing purposes only
def load_traininRating_test():
    global USERID_MOVIE_RATING, MOVIE_USERID, __data_training_loaded
    count = 0
    #userID::movieID::Rating::timestamp
    for line in open(FILE_PATH_TRAININGRATINGS):
        fields = line.strip().split("::")
        user_id = int(fields[0])
        movie_id = int(fields[1])
        rating = float(fields[2])
       
        #Debugging
        print(count)
        count += 1
        ##########

        if user_id in USERID_MOVIE_RATING:
            USERID_MOVIE_RATING[user_id][movie_id] = rating
            #USERID_MOVIE_RATING[user_id].append((movie_id, rating))
        else:
            USERID_MOVIE_RATING[user_id] = {}
            USERID_MOVIE_RATING[user_id][movie_id] = rating

        if movie_id in MOVIE_USERID:
            MOVIE_USERID[movie_id].append(user_id)
        else:
            MOVIE_USERID[movie_id] = [user_id]

        if count > 100000:
            break            
    __data_training_loaded = True



def load_movie_rating():
    global MOVIE_RATING, __data_movie_rating_loaded
    count = 0
    for line in open(FILE_PATH_TRAININGRATINGS):
        fields = line.strip().split("::")
        movie_id = int(fields[1])
        rating = float(fields[2])

        #Debugging
        if count % 1000000 == 0:
            print(count)
        count += 1
        ##########

        if movie_id in MOVIE_RATING:
            MOVIE_RATING[movie_id].append(rating)
        else:
            MOVIE_RATING[movie_id] = [rating]

    __data_movie_rating_loaded = True


# Load all data. This takes a long time but must be done for performance later
# Sets data loaded flag to true
def setup():
    load_movies()
    load_trainingRating()
    #global __data_loaded
    #__data_loaded = True

def setup_avg_rating():
    global __average_rating
    __average_rating = average_movie_rating_over_all_movies()

# OUTPUT METHODS
def format_result(result):
    #format = user_id::movie_id,rating
    if len(result) < 3:
        return ""

    user_id = result[0]
    movie_id = result[1]
    rating = result[2]
    print(user_id, movie_id, rating)
    return "{}::{},{}".format(user_id, movie_id, round(rating,2))

# Format result and appends this in file_handler
# Expects result = [user_id, movie_id, rating]
def store_results_csv(file_handler, result):
    #TODO test file_handler exists
    file_handler.write(format_result(result) + "\n")
    return True


#TODO make class which on setup loads all files into memory as strings 
#or hardcore nice data structs.


# HELPER METHODS

# Parses a string of genres separated by pipe (|).
# Returns list of genres
def parse_genres(genres):
    return genres.split("|")

# Returns genres of the given movie id.
# Constant time if data is loaded
def get_genres(movie_id):
    if __data_movies_loaded:
        return MOVIES_GENRES[movie_id]
    else:
        genres = ""
        for line in open(FILE_PATH_MOVIES):
            fields = line.strip().split("::")
            if int(fields[0]) == movie_id:
                genres = parse_genres(fields[2])
                break
        return genres

# Constant time if data is loaded
def get_movies_with_genre(genre):
    if __data_movies_loaded:
        return GENRES_MOVIES[genre]
    else:
        movies = []
        for line in open(FILE_PATH_MOVIES):
            fields = line.strip().split("::")
            if genre in parse_genres(fields[2]):
                movies.append(int(fields[0]))
        return movies

# Constant time if data is loaded
def get_movies_and_ratings_of_user(user_id):
    if __data_training_loaded:
        return USERID_MOVIE_RATING[user_id]
    else:
        dict_movie_rating = {} #movie -> rating
        for line in open(FILE_PATH_TRAININGRATINGS):
            fields = line.strip().split("::")
            if user_id == int(fields[0]):
                movie_id = int(fields[1])
                rating = int(fields[2])
                dict_movie_rating[movie_id] = rating
        return dict_movie_rating

# Expects line format = user_id::movie_id or user_id::movie_id[::string]
def parse_test_file_line(line):
    fields = line.split("::")
    user_id, movie_id = fields[0], fields[1]
    return [int(user_id), int(movie_id)]

# Expects line format = user_id::movie_id,rating
def parse_result_file_line(line):
    line = line.strip()
    [user_id, rest] = line.split("::")
    [movie_id, result] = rest.split(",")
    return [int(user_id), int(movie_id), float(result)]


# Expects line format = user_id::movie_id::rating::timestamp
def parse_training_file_line(line):
    fields = line.strip().split("::")
    return [int(fields[0]), int(fields[1]), float(fields[2]), int(fields[3])] #should be tuple


# Used for RMSE
def parse_training_result_file_line(line):
    fields = line.strip().split("::")
    return [int(fields[0]), int(fields[1]), float(fields[2])]



# Loads results of two files and returns tuple of lists 
# [[user_id, movie_id, result]] containing all entries of both files sorted 
# on user_id and movie_id.
# Returns ([],[]) if files are empty or they are unaligned
def setup_results(file_handler1, file_handler2, parse_line1, parse_line2):
    #[[user_id, movie_id, rating]]
    list1 = [parse_line1(line) for line in file_handler1 if line.strip() != "" and "Id" not in line.strip()]
    list2 = [parse_line2(line) for line in file_handler2 if line.strip() != "" and "Id" not in line.strip()]

    if len(list1) != len(list2):
        print("Uneven amount of data loaded. Content and Collaborative result files must have same number of entries")
        return ([], [])

    list1 = sorted(list1, key=lambda s: (s[0],s[1]))
    list2 = sorted(list2, key=lambda s: (s[0],s[1]))

    for fields1, fields2 in zip(list1, list2):
        if fields1[0] != fields2[0] or fields1[1] != fields2[1]:
            print("Unaligned data loaded. Found user_id or movie_id not aligned")
            print("user_id1 =",fields1[0], "user_id2 =", fields2[0],  "movie_id1 =", fields1[1], "movie_id2 =",fields2[1])
            return ([], [])

    return (list1, list2)



# Takes two file handlers and computes the RMSE between them, i.e. between the
# test results and the true results.
# Takes a parse function for each file to parse lines.
# Expects both files to start from first line (i.e. no header) and that they 
# have same number of lines/entries. If they do not align, an error will be
# reported.
def RMSE(file_true_result, file_test, parse_true_result, parse_test):
    RMSE = 0
    count = 0
    total = 0
    list_true_result, list_test_result = setup_results(file_true_result, file_test, parse_true_result, parse_test)
    for result_fields, test_fields in zip(list_true_result, list_test_result):
        count += 1
        total += pow(result_fields[2] - test_fields[2], 2)
    if count > 0:
        RMSE = math.sqrt(total/count)
    return RMSE


# BASELINE METHODS

# Baseline method
# Your prediction for a movie should just be the average rating over all movies
def average_movie_rating_over_all_movies():
    total = 0
    count = 0
    result = 0
    for line in open(FILE_PATH_TRAININGRATINGS):
        fields = line.strip().split("::")
        total += float(fields[2])
        count += 1
    if count != 0:
        result = total/count
    return result
        
# Baseline method
# Your prediction for a rating given by a user should just be the average
# rating given by the user
def average_rating_by_user(user_id):
    if __data_training_loaded:
        mov_rat_dict = USERID_MOVIE_RATING[user_id]
        # movie - > rating
        ratings = mov_rat_dict.values()
        if len(ratings) > 0:
            return sum(ratings)/len(ratings)
        else:
            return 0
    else:
        total = 0
        count = 0
        result = 0
        for line in open(FILE_PATH_TRAININGRATINGS):
            fields = line.strip().split("::")
            if int(fields[0]) == user_id:
                total += float(fields[2])
                count += 1
        if count != 0:
            result = total/count
        return result

# Baseline method
# Your prediction for a movie should just be the average rating over all users
# for that movie
# Assuming parameter is movieID not movie name
def average_rating_of_movie_over_all_users(movie_id):
    if __data_movie_rating_loaded:
        ratings = MOVIE_RATING[movie_id]
        if len(ratings) > 0:
            return sum(ratings)/len(ratings)
        else:
            return 0
    else:
        #TODO create dict to use instead of this
        total = 0
        count = 0
        result = 0
        for line in open(FILE_PATH_TRAININGRATINGS):
            fields = line.strip().split("::")
            if int(fields[1]) == movie_id:
                total += float(fields[2])
                count += 1
        if count != 0:
            result = total/count
        return result

def simple1(user_id, movie_id):
    return __average_rating

def simple2(user_id, movie_id):
    return average_rating_by_user(user_id)

def simple3(user_id, movie_id):
    return average_rating_of_movie_over_all_users(movie_id)


# RECOMMENDER SYSTEMS

# Content based approach
# Given userID and movieID, return estimated rating.
# Based on ratings given by the user to similar movies (via genres) in the past
# Returns 2.5 if no estimate is possible
def content_based(user_id, movie_id):
    est_rating = 2.5
    total = 0
    count = 0
    genres = get_genres(movie_id)
    movies = []

    # Find movies of same genre that user has seen in the past
    for genre in genres:
        movies.extend(get_movies_with_genre(genre))

    # Clear original movie
    movies = [m_id for m_id in movies if m_id != movie_id]

    # Filter only movies seen by user
    dict_movie_rating = get_movies_and_ratings_of_user(user_id)
    movies = [m_id for m_id in movies if m_id in dict_movie_rating.keys()]

    # Use similarity degree to determine weighted rating
    # The frequency of a movie determines the degree of the corresponding rating
    for m_id in movies:
        total += dict_movie_rating[m_id]
        count += 1

    if count != 0:
        est_rating = total/count

    return est_rating



# Computes the similarity between the two given dicts of movies and ratings.
# Returns degree of similarity
# Define similarity of users: #sharedMovs/#totalRatedMovs?? 
# Employing Pearson correlation coefficient, 1 = total positive correlation, 
# 0 is no correlation and -1 is total negative correlation
def compute_similarity(dict_movie_rating, dict_target_movie_rating, shared_movie_ids):
    sim = 0
    
    num_shared_movies = len(shared_movie_ids)
    if num_shared_movies <= 0:
        print("ERROR: compute_similarity num_shared_movies is zero")
        return -1
    mean_rating = sum([dict_movie_rating[m_id] for m_id in shared_movie_ids])/num_shared_movies
    mean_target_rating = sum([dict_target_movie_rating[m_id] for m_id in shared_movie_ids])/num_shared_movies
    #similarity_vector = [PCC(dict_movie_rating[m_id], dict_target_movie_rating[m_id]) for m_id in shared_movie_ids]
    # List comprehensions should be a bit faster than loops
    #d_numerator = sum([(dict_movie_rating[m_id]-mean_rating) * (dict_target_movie_rating[m_id]-mean_target_rating) for m_id in shared_movie_ids])
    #print("mean_rating = ", mean_rating, ", mean_target_rating = ", mean_target_rating)
    estimate = 0
    magnitude = 0
    magnitude_target = 0
    for m_id in shared_movie_ids:
        #Compute similarity
        r = dict_movie_rating[m_id]
        r_target = dict_target_movie_rating[m_id]

        estimate += r*r_target
        magnitude += pow(r, 2)
        magnitude_target = pow(r_target, 2)

    #print("magnitude = ", magnitude,", magnitude_target = ", magnitude_target)
    #In case a user always rates the same, the diff will always be 0. Set to 1. 
    if magnitude == 0:
        magnitude = 1
    if magnitude_target == 0:
        magnitude_target = 1

    if magnitude > 0 and magnitude_target > 0:
        sim = estimate/math.sqrt(magnitude)*math.sqrt(magnitude_target)
    else:
        print("ERROR: compute_similarity magnitude or magnitude_target is zero")
        print("magnitude = ", magnitude,", magnitude_target = ", magnitude_target)
    #print("sim = ", sim)
    #DEBUGGING
    #print("similarity degree = ", sim)

    return sim


# Collaborative filtering based approach
# Based on cosine-based approach
def collaborative_filtering_based(user_id, movie_id):
    # USERID_MOVIE_RATING #user_id -> {movie -> rating}
    # MOVIE_USERID #movie_id -> [user_id]
    total_rating = 0
    k = 0

    # Find [movie_id] rated by user_id
    dict_movie_rating = USERID_MOVIE_RATING[user_id]
    # Find all users who has rated movie_id
    target_user_ids = MOVIE_USERID[movie_id]

    for target_user_id in target_user_ids:
        # get movies rated by target_user_id
        dict_target_movie_rating = USERID_MOVIE_RATING[target_user_id]
        shared_movie_ids = set(dict_movie_rating.keys()).intersection(dict_target_movie_rating.keys())
        if len(shared_movie_ids) <= 0:
            continue
        #TODO MAYBE ABSOLUTE SIM
        #sim = abs(compute_similarity(dict_movie_rating, dict_target_movie_rating, shared_movie_ids))
        sim = compute_similarity(dict_movie_rating, dict_target_movie_rating, shared_movie_ids)
        total_rating += sim*dict_target_movie_rating[movie_id]
        k += abs(sim)


    # Average the aggregation as in 10a
    #total_rating = total_rating/len(target_user_ids)
    # 10 b:
    estimated_rating = 0
    #Cannot use k=0 to anything.
    if k == 0:
        k = 1
    if k > 0:
        estimated_rating = total_rating/k
    else:
        print("ERROR: collaborative_filtering_based division by zero")

    #print("total rating for user", user_id, "and movie", movie_id, "=", estimated_rating)

    return abs(estimated_rating)


# Hybrid approach
# 50/50 from content_based and collaborative_filtering_based predictions
def hybrid_based(content_rating, collaborative_rating):
    p_collaborative = 0.55 #0.55
    p_content = 1-p_collaborative #0.45
    return content_rating*p_content + collaborative_rating*p_collaborative



# Experiments


# Combine results of content based and collaborative based computations
# using hybrid_based().
# Expects list_content and list_collaborative are aligned and are on the form
# [[user_id, movie_id, rating]]
def hybrid_compute(list_content, list_collaborative, result_queue):
    for content_fields, collaborative_fields in zip(list_content, list_collaborative):
        rating = hybrid_based(content_fields[2], collaborative_fields[2])
        result_queue.put((content_fields[0], content_fields[1], rating))


def result_thread(file_name_out, result_queue, kill_conn):
    out = open(file_name_out, 'a')
    while(True):
        try:
            (user_id, movie_id, result) = result_queue.get(True, 1) #timeout 1 sec
            store_results_csv(out, [user_id, movie_id, result])
            out.flush()
        except:
            if kill_conn.poll():
                print("result thread closing down")
                break

    out.close()

# Worker thread used by threaded experiment methods
def worker_thread(line, method, result_queue):
    try:
        [user_id, movie_id] = parse_test_file_line(line)
        #print("worker_thread before method")
        result = method(user_id, movie_id)
        #print("worker_thread after method")
        result_queue.put((user_id, movie_id, result))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(traceback.format_exc())
        return False
    return True

# Given file, number of threads, run the method on the file and store results
def run_method_threaded(result_file_name, file, number_workers, method):
    queue_output = Queue()
    (kill_con, not_used) = Pipe()
    # start output process:
    result_p = Process(target=result_thread, args=(result_file_name + ".csv", queue_output, kill_con,))
    result_p.start()
    # start workers worker processes
    #pool = concurrent.futures.ProcessPoolExecutor(number_workers)    
    pool = concurrent.futures.ThreadPoolExecutor(number_workers)
    future = [pool.submit(worker_thread, line, method, queue_output)
        for line in file if line.strip() != "" and line.strip() != "Id"]
    concurrent.futures.wait(future)
    #store_results_csv(out, future.result())
    kill_con.send("die!")
    result_p.join()


def run_method(result_file_name, file, method):
    out = open(result_file_name + ".csv", 'a')
    for line in file:
        if line.strip() != "" and line.strip() != "Id":
            [user_id, movie_id] = parse_test_file_line(line)
            result = method(user_id, movie_id)
            store_results_csv(out, [user_id, movie_id, result])


if __name__ == "__main__":
    import sys
    n = len(sys.argv)
    method_name = ""
    data_set = ""

    if n <= 1:
        sys.exit(0)
    elif n == 2:
        method_name = sys.argv[1]
        data_set = "test"
        test_file_handler = load_test_file()
    else:
        method_name = sys.argv[1]
        data_set = sys.argv[2]
        if data_set == "training":
            test_file_handler = load_training_file()
            file_name = "trainingResultFile"
        elif data_set == "test":
            test_file_handler = load_test_file()
            file_name = "submissionFile"


    # Each thread will use approx double the amount of mem as 1 thread.
    if method_name == "simple1":
        #number_workers = 3
        file_name += "Simple1"
        setup_avg_rating()
        init_submission_file(file_name)
        run_method(file_name, test_file_handler, simple1)
    elif method_name == "simple2":
        #number_workers = 3
        file_name += "Simple2"
        load_trainingRating()
        init_submission_file(file_name)
        #run_method(file_name, test_file_handler, number_workers, simple2)
        run_method(file_name, test_file_handler, simple2)
    elif method_name == "simple3":
        number_workers = 4
        file_name += "Simple3"
        load_movie_rating()
        init_submission_file(file_name)
        #run_method_threaded(file_name, test_file_handler, number_workers, simple3)
        run_method(file_name, test_file_handler, simple3)
    elif method_name == "content":
        number_workers = 1
        file_name += "ContentBased"
        setup()
        init_submission_file(file_name)
        run_method(file_name, test_file_handler, content_based)
        #run_method_threaded(file_name, test_file_handler, number_workers, content_based)
    elif method_name == "collaborative":
        number_workers = 2
        file_name += "CollaborativeBased"
        load_trainingRating()
        init_submission_file(file_name)
        run_method_threaded(file_name, test_file_handler, number_workers, collaborative_filtering_based)
    elif method_name == "hybrid":
        print("Expects both content and collaborative to have been runned alread.")
        if data_set == "test":
            file_name_content = "submissionFileContentBased.csv"
            file_name_collaborative = "submissionFileCollaborativeBased.csv"
        else:
            file_name_content = "trainingResultFileContentBased.csv"
            file_name_collaborative = "trainingResultFileCollaborativeBased.csv"
        file_handler_content = open(file_name_content)
        file_handler_collaborative = open(file_name_collaborative)

        file_name += "HybridBased"
        init_submission_file(file_name)
        queue_output = Queue()
        (kill_con, not_used) = Pipe()
        # start output process:
        result_p = Process(target=result_thread, args=(file_name + ".csv", queue_output, kill_con,))
        result_p.start()

        list_content, list_collaborative = setup_results(file_handler_content, file_handler_collaborative, parse_result_file_line, parse_result_file_line)
        if len(list_content) > 0:
            hybrid_compute(list_content, list_collaborative, queue_output)
        #something goes wrong here
        #queue_output.join()
        kill_con.close()
        print("Threads told to close down")
        result_p.join()
    else:
        print("No allowed method given. Exiting")
        sys.exit(0)

    if data_set == "training":
        print("Training data set computed, computing RMSE")
        file_result = open(file_name + ".csv")
        #remove header from result file
        line_header = file_result.readline()
        file_train = load_training_file()
        e = RMSE(file_train, file_result, parse_training_file_line, parse_result_file_line)
        print("RMSE on " + data_set + "using method " + method_name + " = ", e)
    else:
        print("Done. Exiting")
        sys.exit()