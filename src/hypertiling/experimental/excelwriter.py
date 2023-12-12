def write_excel(data, path, t):
    workbook = xlsxwriter.Workbook(path)  # writing to .csv instead might be faster
    ws = workbook.add_worksheet()

    # Bring border sites to the top of the matrix and sort each part by polygon number
    # pos = 0
    # for ind, row in enumerate(data):  # border sites are sorted by construction
    #     if np.count_nonzero(row == 0) != 0:
    #         data[[pos, ind]] = data[[ind, pos]]  # swaps the rows at index ind and pos
    #         pos += 1
    #
    # dataslice = data[pos:, :]
    # dataslice = sorted(dataslice, key=lambda dataslice_entry: dataslice_entry[0])  # sorting not-border sites
    # data[pos:, :] = np.stack(dataslice, axis=0)  # sorts the non-border sites

    # Header/first line of the table
    if t == "nn":
        ws.write(0, 0, "Polygon #")  # first column
        ws.write(0, 1, f"x_center")
        ws.write(0, 2, f"y_center")
        for x in range(1, data.shape[1]-2):
            ws.write(0, x+2, f"NN #{x}")

    # write array to table
    #cblue = workbook.add_format({'bg_color': '#8EA9DB'})  # blue for border sites NN < p
    #corange = workbook.add_format({'bg_color': '#F4B084'})  # orange for inner sites NN = p
    for i in range(len(data)):
        for j in range(data.shape[1]):
            if t == "nn":
                ws.write(i + 1, j, data[i, j])
            elif t == "adj":
                ws.write(i, j, data[i, j])
            # if i <= pos - 1:
            #     ws.write(i + 1, j, data[i, j], cblue)
            # else:
            #     ws.write(i + 1, j, data[i, j], corange)

    workbook.close()  # necessary for saving the changes