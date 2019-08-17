# -*- coding: utf-8 -*-

import numpy as np

partial_soln = []
final_soln = []
done = 0

def solve(board, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (may be rotated or flipped), and (rowi, coli) is
    the coordinate of the upper left corner of pi in the board (lowest row and column index
    that the tile covers).

    -Use np.flip and np.rot90 to manipulate pentominos.

    -You can assume there will always be a solution.
    """
    big_matrix = solver_helper(board, pents)
    saved_matrix = big_matrix[:]
    recurse_matrix(big_matrix, list(range(len(big_matrix[0]))), list(range(len(big_matrix))))
    return decode_final_soln(saved_matrix, pents, board)


def decode_final_soln(big_matrix, pents, board):
    global final_soln
    result = []

    for matrix_row in final_soln:
        list_of_ones = []
        for y in range(len(big_matrix[0])):
            if big_matrix[matrix_row][y]:
                if y < len(pents):
                    list_of_ones.append(y)
                else:
                    list_of_ones.append(y - len(pents))

        # Pentomino is first 1 in list
        pent = list_of_ones[0]
        del list_of_ones[0]

        smallest_x = 5000
        smallest_y = 5000
        coords = []
        # Convert remaining ones into coordinates -- WORKS
        for space in list_of_ones:
            x = (int)(space%len(board[0]))
            y = (int)(space/len(board[0]))
            if x < smallest_x:
                smallest_x = x;
            if y < smallest_y:
                smallest_y = y;
            # A list of all coordinates in pentomino
            coords.append((x,y))

        #print ("Original:", coords)

        # New coords correct --- changes coords
        greatest_x = -1
        greatest_y = -1
        end = len(coords)
        for c in range(end):
            temp = coords.pop(0)
            new_coord = (temp[0]-smallest_x, temp[1]-smallest_y)
            if new_coord[0] > greatest_x:
                greatest_x = new_coord[0]
            if new_coord[1] > greatest_y:
                greatest_y = new_coord[1]
            coords.append(new_coord)
        # print("New:", coords) # works

        # coords now holds all the new_coords
        chopped_pent = []
        # print("greatest_x:", greatest_x)
        # print("greatest_y:", greatest_y)
        for y in range(greatest_y+1):
            new_row = []
            for x in range(greatest_x+1):
               if (x,y) in coords:
                   new_row.append(pent+1)
               else:
                   new_row.append(0)
            chopped_pent.append(new_row)
        chopped_pent = np.array(chopped_pent)
        result.append((chopped_pent, (smallest_y, smallest_x)));

        #print("------")
   # print (result)
    return result




def recurse_matrix(big_matrix, H, B):
    global partial_soln, final_soln, done
    all_rows = []

    # If H is empty, print solution and return
    if len(H) == 0:
        final_soln = partial_soln[:]
        done = 1
        #print("done")
        return

    if len(H) and not len(big_matrix):
        return

    # Choose a column min_col
    min_col = find_min_col(big_matrix)
    # Find all rows such that column = 1
    for r in range(len(big_matrix)):
        if (big_matrix[r][min_col]):
            all_rows.append(r)

    # For each of these rows
    for chosen_row in all_rows:
        # Add row to partial solution
        partial_soln.append(B[chosen_row])
        # Save state of matrix and column list
        saved_matrix = big_matrix[:]
        saved_H = H[:]
        saved_B = B[:]
        # Find all columns and rows to delete for chosen row
        cols_to_pop = []
        rows_to_pop = []
        for j in range(len(big_matrix[0])):
            if big_matrix[chosen_row][j] == 1:
                cols_to_pop.append(j)
                for i in range(len(big_matrix)):
                    if big_matrix[i][j]:
                        rows_to_pop.append(i)

        # Delete all of them
        rows_to_pop = list(set(rows_to_pop))
        rows_to_pop.sort(reverse = True)
        cols_to_pop.sort(reverse = True)
        for row in rows_to_pop:
            big_matrix.pop(row)
            B.pop(row)
        for col in cols_to_pop:
            for row in range(len(big_matrix)):
                big_matrix[row] = np.delete(big_matrix[row], col)
            H.pop(col)

        # Recursive call
        recurse_matrix(big_matrix, H, B)
        if (done):
            break
        # Restore state of matrix and column list
        B = saved_B[:]
        H = saved_H[:]
        big_matrix = saved_matrix[:]
        # Remove chosen_row from partial solution
        partial_soln.remove(B[chosen_row])


def find_min_col(big_matrix):
    min_col = -1
    min_value = 50000
    for c in range(len(big_matrix[0])): # crashing
        num_ones = 0
        for r in big_matrix:
            num_ones += r[c]
        if num_ones < min_value:
            min_value = num_ones
            min_col = c
    return min_col


def solver_helper(board, pents):
    big_matrix = []
    list_of_squares = []
    new_row = []
    omino_length = len(pents)
    squares_total = len(board)*len(board[0])
    width = len(board[0])
    height = len(board)
    x = omino_length+squares_total
    #print(omino_length, squares_total, x)

    for p in range(omino_length):
        for c in range(width):
            for r in range(height):
                for rot in range(4):
                    rotated_pent = np.rot90(pents[p], rot)
                    if (check_bounds(rotated_pent, board, r, c)):
                        #rotations
                        list_of_squares = get_list_of_squares(rotated_pent, c, r)
                        new_row = fill_matrix(p, list_of_squares, pents, board)
                        if not any(np.array_equal(x, new_row) for x in big_matrix):
                            big_matrix.append(new_row)
                    flipped_pent = np.flip(rotated_pent, 0)
                    if (check_bounds(flipped_pent, board, r, c)):
                        #flips
                        list_of_squares = get_list_of_squares(flipped_pent, c, r)
                        new_row = fill_matrix(p, list_of_squares, pents, board)
                        if not any(np.array_equal(x, new_row) for x in big_matrix):
                            big_matrix.append(new_row)
    return big_matrix


def check_bounds(pent, board, row, col):
    pent_width = len(pent[0])
    pent_height = len(pent)
    # Checks out of outer bounds
    if ((pent_height+row > len(board)) or (pent_width+col > len(board[0]))):
        return False
    # Checks any inner bounds (chess board)
    for c in range(pent_width):
        for r in range(pent_height):
            if (board[row+r][col+c] <= 0 and pent[r][c]):
                return False
    return True

def get_list_of_squares(pent, col, row):
    list_of_squares = []
    pent_width = len(pent[0])
    pent_height = len(pent)
    for c in range(pent_width):
        for r in range(pent_height):
            if(pent[r][c]):
                list_of_squares.append((row+r, col+c))
    return list_of_squares

def fill_matrix(pent, list_of_squares, pents, board): # returns a row to be added to the matrix

    num_of_pents = len(pents)
    squares_total = len(board)*len(board[0])
    row_size = num_of_pents+squares_total
    matrix_row = np.zeros(row_size)
    conversion_arr = []

    for (x, y) in list_of_squares:
        conversion_arr.append(x*len(board[0])+y)
    matrix_row[pent] = 1
    for z in conversion_arr:
        matrix_row[z+num_of_pents] = 1

    return matrix_row
