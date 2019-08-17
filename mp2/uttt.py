from time import sleep
from math import inf
from random import randint

class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        self.maxPlayer='X'
        self.minPlayer='O'
        self.maxDepth=3
        #The start indexes of each local board
        self.globalIdx=[(0,0),(0,3),(0,6),(3,0),(3,3),(3,6),(6,0),(6,3),(6,6)]

        #Start local board index for reflex agent playing
        # self.startBoardIdx=4
        self.startBoardIdx=randint(0,8)

        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility=10000
        self.twoInARowMaxUtility=500
        self.preventThreeInARowMaxUtility=100
        self.cornerMaxUtility=30

        self.winnerMinUtility=-10000
        self.twoInARowMinUtility=-100
        self.preventThreeInARowMinUtility=-500
        self.cornerMinUtility=-30

        self.expandedNodes=0
        self.currPlayer=True
        self.bestMove=(0,0)
        self.design = 0

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]])+'\n')


    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        #print("inside evaluate")

        if (isMax): # max player
            symbol = self.maxPlayer
            other_sym = self.minPlayer
            winner = self.winnerMaxUtility
            two = self.twoInARowMaxUtility
            prevent = self.preventThreeInARowMaxUtility
            corner = self.cornerMaxUtility
        else: # min player
            symbol = self.minPlayer
            other_sym = self.maxPlayer
            winner = self.winnerMinUtility
            two = self.twoInARowMinUtility
            prevent = self.preventThreeInARowMinUtility
            corner = self.cornerMinUtility

        # second rule declarations
        score_rule_2 = 0
        s_sym = set([symbol, '_'])
        s_other_sym = set([other_sym, symbol])

        # first rule
        for boardrow in range(0,3): # 9 local boards
            for boardcol in range(0,3):
                # diags
                if self.board[boardrow*3+1][boardcol*3+1] == symbol: # center spot is taken
                    # first rule
                    if (self.board[boardrow*3][boardcol*3] == symbol and \
                       self.board[boardrow*3+2][boardcol*3+2] == symbol) or \
                       (self.board[boardrow*3+2][boardcol*3] == symbol and \
                       self.board[boardrow*3][boardcol*3+2] == symbol):
                       return winner
                    # second rule
                    temp_set = set([self.board[boardrow*3][boardcol*3], self.board[boardrow*3+2][boardcol*3+2]])
                    if (temp_set == s_sym):
                        score_rule_2 += two
                    temp_set = set([self.board[boardrow*3+2][boardcol*3], self.board[boardrow*3][boardcol*3+2]])
                    if (temp_set == s_sym):
                        score_rule_2 += two
                    # ummmmmm
                    if (self.board[boardrow*3][boardcol*3] == other_sym and self.board[boardrow*3+2][boardcol*3+2] == other_sym):
                        score_rule_2 += prevent
                    if (self.board[boardrow*3+2][boardcol*3] == other_sym and self.board[boardrow*3][boardcol*3+2] == other_sym):
                        score_rule_2 += prevent

                # ummmmm
                if self.board[boardrow*3+1][boardcol*3+1] == '_':
                    if (self.board[boardrow*3][boardcol*3] == symbol and \
                       self.board[boardrow*3+2][boardcol*3+2] == symbol):
                        score_rule_2 += two
                    if (self.board[boardrow*3+2][boardcol*3] == symbol and \
                       self.board[boardrow*3][boardcol*3+2] == symbol):
                       score_rule_2 += two

                # second rule
                if self.board[boardrow*3+1][boardcol*3+1] == other_sym: # center spot by opponent
                    temp_set = set([self.board[boardrow*3][boardcol*3], self.board[boardrow*3+2][boardcol*3+2]])
                    if (temp_set == s_other_sym):
                        score_rule_2 += prevent
                    temp_set = set([self.board[boardrow*3+2][boardcol*3], self.board[boardrow*3][boardcol*3+2]])
                    if (temp_set == s_other_sym):
                        score_rule_2 += prevent

                # rows and cols
                numcol = [0]*9
                numcol_two = [0]*9
                numcol_other = [0]*9
                numcol_other_two = [0]*9
                for localrow in range(boardrow*3, boardrow*3+3):
                    numrow = 0
                    numrow_two = 0
                    numrow_other = 0
                    numrow_other_two = 0
                    for localcol in range(boardcol*3, boardcol*3+3):
                        if self.board[localrow][localcol] == symbol:
                            numrow += 1
                            numcol[localcol] += 1
                        # second rule
                        if self.board[localrow][localcol] == other_sym:
                            numrow_other += 1
                            numcol_other[localcol] += 1
                        # first rule
                        if numcol[localcol] == 3 or numrow == 3:
                            return winner
                        # second rule
                        if numrow == 2 and '_' in [self.board[localrow][i] for i in range(boardcol*3,boardcol*3+3)] and numrow_two == 0:                                 # center and non blocked
                                score_rule_2 += two
                                numrow_two = 1
                        if numcol[localcol] == 2 and '_' in [self.board[i][localcol] for i in range(boardrow*3, boardrow*3+3)] \
                            and numcol_two[localcol] == 0:
                                score_rule_2 += two
                                numcol_two[localcol] = 1

                        # prevention
                        if numrow_other == 2 and symbol in [self.board[localrow][i] for i in range(boardcol*3,boardcol*3+3)] and numrow_other_two == 0:                                 # center and non blocked
                                score_rule_2 += prevent
                                numrow_other_two = 1
                        if numcol_other[localcol] == 2 and symbol in [self.board[i][localcol] for i in range(boardrow*3, boardrow*3+3)] \
                            and numcol_other_two[localcol] == 0:
                                score_rule_2 += prevent
                                numcol_other_two[localcol] = 1

        if score_rule_2 != 0:
            return score_rule_2

        # third rule
        score_rule_3 = 0
        corners_idx = [0, 2, 3, 5, 6, 8]
        for row in corners_idx:
            for col in corners_idx:
                if self.board[row][col] == symbol:
                    score_rule_3 += corner

        return score_rule_3


    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0

        if isMax: # offensive agent
            return self.evaluatePredifined(isMax)

        symbol = self.minPlayer
        other_sym = self.maxPlayer
        winner = -1000
        they_win = 10000
        two = -500
        other_two = 400
        prevent = -500
        other_prevent = 400

        # second rule declarations
        s_sym = set([symbol, '_'])
        s_sym_other = set([other_sym, '_'])
        s_other_sym = set([other_sym, symbol])

        # first rule
        for boardrow in range(0,3): # 9 local boards
            for boardcol in range(0,3):
                # diags
                if self.board[boardrow*3+1][boardcol*3+1] == symbol: # center spot is taken
                    # first rule
                    if (self.board[boardrow*3][boardcol*3] == symbol and \
                       self.board[boardrow*3+2][boardcol*3+2] == symbol) or \
                       (self.board[boardrow*3+2][boardcol*3] == symbol and \
                       self.board[boardrow*3][boardcol*3+2] == symbol):
                       score += winner
                    # second rule
                    temp_set = set([self.board[boardrow*3][boardcol*3], self.board[boardrow*3+2][boardcol*3+2]])
                    if (temp_set == s_sym):
                        score += two
                    temp_set = set([self.board[boardrow*3+2][boardcol*3], self.board[boardrow*3][boardcol*3+2]])
                    if (temp_set == s_sym):
                        score += two
                    # ummmmmm
                    if (self.board[boardrow*3][boardcol*3] == other_sym and self.board[boardrow*3+2][boardcol*3+2] == other_sym):
                        score += prevent
                    if (self.board[boardrow*3+2][boardcol*3] == other_sym and self.board[boardrow*3][boardcol*3+2] == other_sym):
                        score += prevent

                # ummmmm
                if self.board[boardrow*3+1][boardcol*3+1] == '_':
                    if (self.board[boardrow*3][boardcol*3] == symbol and \
                       self.board[boardrow*3+2][boardcol*3+2] == symbol):
                        score += two
                    if (self.board[boardrow*3+2][boardcol*3] == symbol and \
                       self.board[boardrow*3][boardcol*3+2] == symbol):
                       score += two
                    if (self.board[boardrow*3][boardcol*3] == other_sym and \
                       self.board[boardrow*3+2][boardcol*3+2] == other_sym):
                        score += other_two
                    if (self.board[boardrow*3+2][boardcol*3] == other_sym and \
                       self.board[boardrow*3][boardcol*3+2] == other_sym):
                       score += other_two

                # second rule
                if self.board[boardrow*3+1][boardcol*3+1] == other_sym: # center spot by opponent
                    temp_set = set([self.board[boardrow*3][boardcol*3], self.board[boardrow*3+2][boardcol*3+2]])
                    if (temp_set == s_other_sym):
                        score += prevent
                    temp_set = set([self.board[boardrow*3+2][boardcol*3], self.board[boardrow*3][boardcol*3+2]])
                    if (temp_set == s_other_sym):
                        score += prevent
                    # they win
                    if (self.board[boardrow*3][boardcol*3] == other_sym and \
                       self.board[boardrow*3+2][boardcol*3+2] == other_sym) or \
                       (self.board[boardrow*3+2][boardcol*3] == other_sym and \
                       self.board[boardrow*3][boardcol*3+2] == other_sym):
                       score += they_win

                    temp_set = set([self.board[boardrow*3][boardcol*3], self.board[boardrow*3+2][boardcol*3+2]])
                    if (temp_set == s_sym_other):
                        score += other_two
                    temp_set = set([self.board[boardrow*3+2][boardcol*3], self.board[boardrow*3][boardcol*3+2]])
                    if (temp_set == s_sym_other):
                        score += other_two

                    if (self.board[boardrow*3][boardcol*3] == symbol and self.board[boardrow*3+2][boardcol*3+2] == symbol):
                        score += other_prevent
                    if (self.board[boardrow*3+2][boardcol*3] == symbol and self.board[boardrow*3][boardcol*3+2] == symbol):
                        score += other_prevent

                # rows and cols
                numcol = [0]*9
                numcol_two = [0]*9
                numcol_other = [0]*9
                numcol_other_two = [0]*9
                for localrow in range(boardrow*3, boardrow*3+3):
                    numrow = 0
                    numrow_two = 0
                    numrow_other = 0
                    numrow_other_two = 0
                    for localcol in range(boardcol*3, boardcol*3+3):
                        if self.board[localrow][localcol] == symbol:
                            numrow += 1
                            numcol[localcol] += 1
                        # second rule
                        if self.board[localrow][localcol] == other_sym:
                            numrow_other += 1
                            numcol_other[localcol] += 1
                        # first rule
                        if numcol[localcol] == 3 or numrow == 3:
                            score += winner
                        if numcol[localcol] == 3 or numrow == 3:
                            score += they_win
                        # second rule
                        if numrow == 2 and '_' in [self.board[localrow][i] for i in range(boardcol*3,boardcol*3+3)] and numrow_two == 0:                                 # center and non blocked
                                score += two
                                numrow_two = 1
                        if numcol[localcol] == 2 and '_' in [self.board[i][localcol] for i in range(boardrow*3, boardrow*3+3)] \
                            and numcol_two[localcol] == 0:
                                score += two
                                numcol_two[localcol] = 1
                        if numrow_other == 2 and '_' in [self.board[localrow][i] for i in range(boardcol*3,boardcol*3+3)] and numrow_other_two == 0:                                 # center and non blocked
                                score += other_two
                                numrow_other_two = 1
                        if numcol_other[localcol] == 2 and '_' in [self.board[i][localcol] for i in range(boardrow*3, boardrow*3+3)] \
                            and numcol_other_two[localcol] == 0:
                                score += other_two
                                numcol_other_two[localcol] = 1

                        # prevention
                        if numrow_other == 2 and symbol in [self.board[localrow][i] for i in range(boardcol*3,boardcol*3+3)] and numrow_other_two == 0:                                 # center and non blocked
                                score += prevent
                                numrow_other_two = 1
                        if numcol_other[localcol] == 2 and symbol in [self.board[i][localcol] for i in range(boardrow*3, boardrow*3+3)] \
                            and numcol_other_two[localcol] == 0:
                                score += prevent
                                numcol_other_two[localcol] = 1
                        if numrow == 2 and other_sym in [self.board[localrow][i] for i in range(boardcol*3,boardcol*3+3)] and numrow_two == 0:                                 # center and non blocked
                                score += other_prevent
                                numrow_two = 1
                        if numcol[localcol] == 2 and other_sym in [self.board[i][localcol] for i in range(boardrow*3, boardrow*3+3)] \
                            and numcol_two[localcol] == 0:
                                score += other_prevent
                                numcol_two[localcol] = 1

        return score

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        #YOUR CODE HERE
        for i in range(0,9):
            for j in range(0,9):
                if self.board[i][j] == '_': # empty space
                    return True

        return False

    def checkWinner(self):
        #Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        #YOUR CODE HERE
        for boardrow in range(0,3): # 9 local boards
            for boardcol in range(0,3):
                # diags
                if self.board[boardrow*3+1][boardcol*3+1] == self.maxPlayer: # center spot is taken
                    if (self.board[boardrow*3][boardcol*3] == self.maxPlayer and \
                       self.board[boardrow*3+2][boardcol*3+2] == self.maxPlayer) or \
                       (self.board[boardrow*3+2][boardcol*3] == self.maxPlayer and \
                       self.board[boardrow*3][boardcol*3+2] == self.maxPlayer):
                       return 1
                if self.board[boardrow*3+1][boardcol*3+1] == self.minPlayer: # center spot is taken
                    if (self.board[boardrow*3][boardcol*3] == self.minPlayer and \
                       self.board[boardrow*3+2][boardcol*3+2] == self.minPlayer) or \
                       (self.board[boardrow*3+2][boardcol*3] == self.minPlayer and \
                       self.board[boardrow*3][boardcol*3+2] == self.minPlayer):
                       return -1

                # rows and cols
                numcol_max = [0]*9
                numcol_min = [0]*9
                for localrow in range(boardrow*3, boardrow*3+3):
                    numrow_max = 0
                    numrow_min = 0
                    for localcol in range(boardcol*3, boardcol*3+3):
                        if self.board[localrow][localcol] == self.maxPlayer:
                            numrow_max += 1
                            numcol_max[localcol] += 1
                        if numcol_max[localcol] == 3 or numrow_max == 3:
                            return 1
                        if self.board[localrow][localcol] == self.minPlayer:
                            numrow_min += 1
                            numcol_min[localcol] += 1
                        if numcol_min[localcol] == 3 or numrow_min == 3:
                            return -1

        return 0

    def alphabeta(self,depth,currBoardIdx,alpha,beta,isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        bestValue=0.0
        bestmoo = (0,0)

        if depth == 3:
            if self.design:
                return self.evaluateDesigned(not isMax)
            else:
                return self.evaluatePredifined(not isMax)

        self.expandedNodes += 1

        currBoardRow = int(currBoardIdx / 3)
        currBoardCol = currBoardIdx % 3

        aa = alpha
        bb = beta

        if isMax: # max
            symbol = self.maxPlayer
            bestValue = -100000
        else:
            symbol = self.minPlayer
            bestValue = 100000

        # loop over local board
        for row in range(0, 3):
            for col in range(0, 3):
                if isMax:
                    aa = max(bestValue, aa)
                else:
                    bb = min(bestValue, bb)
                if self.board[currBoardRow*3+row][currBoardCol*3+col] == '_':
                    self.board[currBoardRow*3+row][currBoardCol*3+col] = symbol
                    val = self.alphabeta(depth+1, row*3+col, aa, bb, not isMax)
                    self.board[currBoardRow*3+row][currBoardCol*3+col] = '_'
                    if isMax and val > bestValue:
                        bestValue = val
                        bestmoo = (currBoardRow*3+row,currBoardCol*3+col)
                    elif not isMax and val < bestValue:
                        bestValue = val
                        bestmoo = (currBoardRow*3+row,currBoardCol*3+col)
                if isMax and bestValue >= bb:
                    return bestValue
                elif not isMax and bestValue <= aa:
                    return bestValue

        self.bestMove = bestmoo
        return bestValue

    def minimax(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """

        bestmoo = (0,0)

        if depth == 3:
            return self.evaluatePredifined(not isMax)

        self.expandedNodes += 1

        if isMax: # max
            symbol = self.maxPlayer
            bestValue = -100000
        else:
            symbol = self.minPlayer
            bestValue = 100000

        currBoardRow = int(currBoardIdx / 3)
        currBoardCol = currBoardIdx % 3

        # loop over local board
        for row in range(0, 3):
            for col in range(0, 3):
                if self.board[currBoardRow*3+row][currBoardCol*3+col] == '_':
                    self.board[currBoardRow*3+row][currBoardCol*3+col] = symbol
                    val = self.minimax(depth+1, row*3+col, not isMax)
                    self.board[currBoardRow*3+row][currBoardCol*3+col] = '_'
                    if isMax and val > bestValue:
                        bestValue = val
                        bestmoo = (currBoardRow*3+row,currBoardCol*3+col)
                    elif not isMax and val < bestValue:
                        bestValue = val
                        bestmoo = (currBoardRow*3+row,currBoardCol*3+col)

        self.bestMove = bestmoo
        return bestValue

    def playGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        bestValue=[]
        gameBoards=[self.board[:][:]]
        expandedNodes=[]

        currentIdx = 4
        if maxFirst:
            isMax = True
        else:
            isMax = False

        while self.checkWinner() == 0 and self.checkMovesLeft():
            self.printGameBoard()
            if isMax:
                symbol = self.maxPlayer
                if isMinimaxOffensive:
                    bestval = self.minimax(0, currentIdx, isMax)
                else:
                    bestval = self.alphabeta(0, currentIdx, -100000, 100000, isMax)
            else:
                symbol = self.minPlayer
                if isMinimaxDefensive:
                    bestval = self.minimax(0, currentIdx, isMax)
                else:
                    bestval = self.alphabeta(0, currentIdx, -100000, 100000, isMax)
            self.board[self.bestMove[0]][self.bestMove[1]] = symbol
            bestMove.append((self.bestMove[0], self.bestMove[1]))
            expandedNodes.append(self.expandedNodes + 0)
            self.expandedNodes = 0
            bestValue.append(bestval)
            gameBoards.append(self.board[:][:])
            isMax = not isMax
            currentIdx = (self.bestMove[0] % 3)*3 + (self.bestMove[1] % 3)

        self.printGameBoard()
        return gameBoards, bestMove, expandedNodes, bestValue, self.checkWinner()

    def playGameYourAgent(self):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        expandedNodes=[]
        winner=0

        self.design = 1

        currentIdx = self.startBoardIdx
        isMax = bool(randint(0,1))

        while self.checkWinner() == 0 and self.checkMovesLeft():
            self.printGameBoard()
            bestval = self.alphabeta(0, currentIdx, -100000, 100000, isMax)
            if isMax:
                symbol = self.maxPlayer
            else:
                symbol = self.minPlayer
            self.board[self.bestMove[0]][self.bestMove[1]] = symbol
            bestMove.append((self.bestMove[0], self.bestMove[1]))
            expandedNodes.append(self.expandedNodes + 0)
            self.expandedNodes = 0
            gameBoards.append(self.board[:][:])
            isMax = not isMax
            currentIdx = (self.bestMove[0] % 3)*3 + (self.bestMove[1] % 3)

        self.printGameBoard()
        summ = 0
        for x in expandedNodes:
            summ += x
        print("sum: ", summ)
        print("list: ", expandedNodes)
        return gameBoards, bestMove, self.checkWinner()


    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0

        self.design = 1

        isMax = False
        currentIdx = self.startBoardIdx

        while self.checkWinner() == 0 and self.checkMovesLeft():
            currBoardRow = int(currentIdx / 3)
            currBoardCol = currentIdx % 3
            self.printGameBoard()
            if isMax:
                symbol = self.maxPlayer
                while True:
                    datain = input("Please enter your move as row col (separated by spaces): ")
                    datain = datain.split()
                    self.bestMove = (int(datain[0]), int(datain[1]))
                    if self.bestMove[0] >= currBoardRow*3 and self.bestMove[0] <= currBoardRow*3+2 \
                        and self.bestMove[1] >= currBoardCol*3 and self.bestMove[1] <= currBoardCol*3+2:
                            break
                    print("Not a valid move. Try again.")
            else:
                symbol = self.minPlayer
                bestval = self.alphabeta(0, currentIdx, -100000, 100000, isMax)
            self.board[self.bestMove[0]][self.bestMove[1]] = symbol
            bestMove.append((self.bestMove[0], self.bestMove[1]))
            gameBoards.append(self.board[:][:])
            isMax = not isMax
            currentIdx = (self.bestMove[0] % 3)*3 + (self.bestMove[1] % 3)

        self.printGameBoard()
        return gameBoards, bestMove, self.checkWinner()

if __name__=="__main__":
    uttt=ultimateTicTacToe()
    # gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(True,False,False)
    gameBoards, bestMove, winner = uttt.playGameYourAgent()
    # summ = 0
    # for x in expandedNodes:
    #     summ += x
    # print("sum: ", summ)
    # print("list: ", expandedNodes)

    # gaemBoards, bestMove, winner = uttt.playGameHuman()

    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
