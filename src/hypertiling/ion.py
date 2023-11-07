# this file contains the logging settings of HYPERTILINGS's printed messages

GLOBAL_VERBOSITY = "Warning"

VERBOSITY_LEVELS = {"Warning": 1, 
                    "Status": 2,
                    "Debug": 3,
                    "Develop": 4}


def show_verbosity_level():
    """
    Display current verbosity level
    """
    print("[hypertiling] The verbosity level is set to:")
    for key,item in VERBOSITY_LEVELS.items():
        arrow = "     "
        if key==GLOBAL_VERBOSITY:
            arrow = " >>> "
        print(arrow+str(item),key)    


def set_verbosity_level(verbosity_depth="Warning"):
    """
    Set verbosity level globally
    """
    if verbosity_depth not in VERBOSITY_LEVELS:
        raise ValueError("[hypertiling] Error: Verbosity level not supported. Select one of the following: "+str(list(VERBOSITY_LEVELS.keys())))
    else:
        global GLOBAL_VERBOSITY
        GLOBAL_VERBOSITY = verbosity_depth
    show_verbosity_level()


def htprint(verbosity_depth, message):
    """
    Use this function for print messages

    Parameters:
        verbosity_depth : str
            defines the verbosity level of that message
        message : str
            the actual message content
    """
    if VERBOSITY_LEVELS[verbosity_depth] <= VERBOSITY_LEVELS[GLOBAL_VERBOSITY]:
        prefix = "[hypertiling] "+verbosity_depth+": "
        print(prefix+message)



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
