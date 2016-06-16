
def convert_to_csv(path):
    file_name_out = "baidu_results_out.csv"
    out = open(file_name_out, 'w+')
    out.write("id,link\n")
    for line in open(path):
        fields = line.strip().split(" ")
        link_id = fields[2]
        url = fields[5]
        out.write("{},{}\n".format(link_id,url))


if __name__ == "__main__":
    import sys
    #Assumes local file exists
    path = "baidu_results_out.txt"
    convert_to_csv(path)