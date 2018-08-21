# c_chess_minimax_alpha_beta

This project is a chess game that uses minimax and alpha beta pruning to drive its AI players. The chess engine uses FIDE rules.

There is a two player (human vs. human), one player (human vs. computer), and zero player mode (computer vs. computer). Human players play by first choosing a color they want to play. Then every turn during the human player's ply they select the file and rank of a piece they want to move and then the file and rank of where they want to move the piece. This is all plotted on a jpeg board displaying the current state of the game.

For games involving at least one computer, you choose the difficulty of the computer player(s) which corresponds to the number of plies that the computer will go into the minimax alpha-beta game tree.


## Inputs

### knight_position_modifiers.txt
This input file contains an 8x8 matrix that corresponds to the location of knights on the chessboard. Depending on where each knight is on the board, there is a larger reduction in the knights value the farther from the center the knight is during board evaluation.

### non_passed_pawn_values.txt
This input file contains a 16x8 matrix that corresponds to the location of pawns on the chessboard that are not passed. The first 8 rows are used if the game is still in the opening stage and the last 8 rows are used if the game is in the end game stage. The score modifiers are larger for non-passed pawns deeper into the board. However, in the opening stage, the scores are larger closer to the center whereas in the end game stage the larger scores are further to the side edges.

### pawn_advancement_multipliers.txt
This input file contains a 4x4 matrix that corresponds to the location of pawns in the 4th through 7th ranks. The first two rows are used for non-passed pawns and the last two are used for passed pawns. In each subset, the first row is for pawns that aren't connected and the second row is for pawns that are connected. The columns represent the depth into the opposing board, with larger score modifiers corresponding with a deeper rank into enemy territory.


## Outputs

### One & two player games
When there is at least one human player, the board state is displayed in chessboard.jpeg which is plotted from plotscripts/chessboard_basic.gplot which uses just basic shapes or plotscripts/chessboard_artwork.gplot which uses piece artwork.

### Zero player games
When there are no human players, the board state is displayed in images/chessboard_x.jpeg (where x is the number of plies played) which is plotted from plotscripts/chessboard_simulation_basic.gplot which uses just basic shapes or plotscripts/chessboard_simulation_artwork.gplot which uses piece artwork.
