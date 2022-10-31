import csv

def write_csv(fname, nbrs):
    """
    Saves the neighbour list into a CSV table file

    Arguments:
    -----------
    fname : str
        Output file name including directory
    nbrs : List[List[int]]:
            Neighbors list
    """
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(nbrs)
