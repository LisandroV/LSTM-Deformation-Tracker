def parse_line(str_line):
    """Parses dodata from str_line and initializes a control point history."""
    tokens = str_line.split()
    ident = tokens[0]
    birth_time = tokens[1]
    death_time = tokens[2]

    # History
    rest = tokens[3:]
    if rest[0][0] != "[":
        raise Exception("'[' was expected but found " + rest[0] + " instead.")
    if rest[-1][-1] != "]":
        raise Exception("']' was expected but found " + rest[-1] + " instead.")
    # Remove '[' and ']'
    rest[0] = rest[0][1:]
    rest[-1] = rest[-1][:-1]

    hist = []
    iterator = iter(rest)
    x_str = next(iterator, False)
    while x_str:
        x = x_str[1:]
        y = next(iterator)
        prev_neighbour_index = next(iterator)
        next_neighbour_index = next(iterator)[:-1]
        x_str = next(iterator, False)
        hist.append((x, y, prev_neighbour_index, next_neighbour_index))

    return (ident, birth_time, death_time, hist)


def convert_hist_to_csv(hist_file: str, csv_file: str):
    # read .hist
    hist_lines = []
    with open(hist_file, "r") as read_file:
        next(read_file)
        for line in read_file:
            hist_lines.append(parse_line(line))

    # create .csv
    f = open(csv_file, "w")
    csv_header = "id,time_step,birth_time,death_time,x,y,prev_id,next_id\n"
    f.write(csv_header)
    for hist_line in hist_lines:
        for time, control_point in enumerate(hist_line[3]):
            time_step = int(hist_line[1]) + time
            csv_row = [
                hist_line[0],
                str(time_step),
                hist_line[1],
                hist_line[2],
                *control_point,
            ]
            f.write(",".join(csv_row) + "\n")
    f.close()


hist_file = "data/sponge_centre/control_points.hist"
csv_file = "data/sponge_centre/control_points.csv"
convert_hist_to_csv(hist_file, csv_file)
