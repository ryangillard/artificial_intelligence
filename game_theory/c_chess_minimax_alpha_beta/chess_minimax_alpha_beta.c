#include <stdio.h> /* printf, scanf, puts */
#include <stdlib.h> /* realloc, free, exit, NULL */

/******************************************************************************************************/
/**************************************** GLOBAL FLAGS/OPTIONS ****************************************/
/******************************************************************************************************/

/* Whether we want to activate the computer move log file to keep track of all moves tried during minimax */
int activate_computer_move_log = 0;

/* Whether we want to activate the computer move log file to keep track of all move scores tried during minimax */
int activate_computer_move_score_log = 0;

/* Whether we want to activate the board history log file to keep track of the board throughout the game */
int activate_board_history_log = 0;

/* Whether we want to use piece artwork or just simple circles */
int use_actual_piece_artwork = 1;

/* Whether we start with a partial board or start off fresh */
int start_with_partial_board = 0;

/* The number of moves to break a computer player's loop */
int moves_to_break_computer_loop = 0;

/* Whether to use optimized computer moves or not */
int optimized_computer_moves = 1;

/* The maximum depth of the game tree that a computer can search */
unsigned int maximum_depth = 10;

/* Whether we use alpha beta pruning or not */
int use_alpha_beta_pruning = 1;

// Whether we want to put off a quick win or not */
int delay_wins = 0;

/* Whether we want to hope for a win instead of settling for a draw */
int disfavor_draws = 1;

/* Whether we want to put off a quick draw or not */
int delay_draws = 0;

/* The column names of the board */
char column_names[] = {"abcdefgh"};

/* The names of each type of piece */
char piece_type_names[6][8] = {"pawn", "knight", "bishop", "rook", "queen", "king"};

/******************************************************************************************************/
/********************************************* STRUCTURES *********************************************/
/******************************************************************************************************/

/* This structure holds the game constants */
struct ChessConstants
{
	unsigned int number_of_colors;
	unsigned int number_of_rows;
	unsigned int number_of_columns;
	unsigned int number_of_unique_pieces;

	unsigned int max_pieces_per_type[6];

	unsigned int number_of_possible_pawn_moves;
	unsigned int number_of_possible_knight_moves;
	unsigned int number_of_possible_king_moves;
} chess_constants;

/* This structure keeps track of player options */
struct PlayerOptions
{
	unsigned int board_history_length;

	int players;
	int color_choice;

	int difficulty_computer1;
	int difficulty_computer2;
};

/* This structure keeps track of the board state */
struct BoardState
{
	unsigned int turn;

	unsigned int white_pieces;
	unsigned int black_pieces;
	unsigned int white_pieces_old;
	unsigned int black_pieces_old;

	int white_in_check;
	int black_in_check;

	int white_queen_side_castle;
	int black_queen_side_castle;
	int white_king_side_castle;
	int black_king_side_castle;

	int white_queen_side_castle_allow;
	int black_queen_side_castle_allow;
	int white_king_side_castle_allow;
	int black_king_side_castle_allow;

	int en_passant_captured;

	unsigned int pawn_move;
	unsigned int capture_move;

	int closed_position;
	int end_game;

	int evaluation_score;
	int game_over;
};

/******************************************************************************************************/
/********************************************* PROTOTYPES *********************************************/
/******************************************************************************************************/

/******************************************************************************************************/
/******************************************** BOARD SETUP *********************************************/
/******************************************************************************************************/

/* This function initializes chess constants */
void InitializeChessConstants();

/* This function initializes the player chosen options */
void InitializePlayerOptions(struct PlayerOptions *player_options);

/* This function sets up the board for play */
void BoardSetup(struct BoardState *board_state, char *basic_chessboard, int ***square, int ***color);

/* This function initializes the board state variables */
void InitializeBoardStateVariables(struct BoardState *board_state);

/* This function setups the initial board */
void InitalBoardSetup(int **square, int **color);

/* This function setups a partial board */
void PartialBoardSetup(int **square, int **color);

/******************************************************************************************************/
/******************************************** PIECE SETUP *********************************************/
/******************************************************************************************************/

/* This function sets up piece counts and position and pawn en passant status */
void PieceSetup(struct BoardState *board_state, int **square, unsigned int ***piece_count, int *****piece_positions, int ***pawn_en_passant_status);

/* This function counts the number of pieces */
void CountPieces(struct BoardState *board_state, int **square, unsigned int **piece_count, int ****piece_positions);

/* This function prints the pieces onto the board */
void PrintPieces(int turn, int **square, int **color, int players, char *basic_chessboard);

/******************************************************************************************************/
/******************************************** PIECE VALUES ********************************************/
/******************************************************************************************************/

/* Initialize piece and position values to be used during board evaluation */
void InitializePieceAndPositionValues(int **piece_value, int ***non_passed_pawn_opening_values, int ***non_passed_pawn_end_game_values, double ***pawn_advancement_multipliers, double ***knight_position_modifiers);

/******************************************************************************************************/
/********************************************* GAME SETUP *********************************************/
/******************************************************************************************************/

/* This function sets up some of the initial game tracking arrays */
void GameSetup(int board_history_length, int **square, int **piece_coordinates, int **move_coordinates, int ****board_history);

/******************************************************************************************************/
/******************************************* BEGIN PLAYING ********************************************/
/******************************************************************************************************/

/* This function gets the number of players from user */
void GetNumberOfPlayers(struct PlayerOptions *player_options);

/******************************************************************************************************/
/****************************************** ADD COMPUTER(S) *******************************************/
/******************************************************************************************************/

/* This function gets the computer's difficulty from the user */
void GetComputerDifficulty(unsigned int maximum_depth, int computer, int *difficulty_computer);

/* This function gets the color choice from the user for single player games */
void GetSinglePlayerColorChoice(struct PlayerOptions *player_options);

/* This function creates computer only variables for logging */
void CreateComputerOnlyLoggingVariables(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, unsigned int maximum_depth, int *depth_daughters, int *depth_valid_move_count, int ***piece_board_history, int ***move_board_history);

/******************************************************************************************************/
/********************************************* NEW TURN ***********************************************/
/******************************************************************************************************/

/* This function starts a new turn */
void StartNewTurn(FILE *outfile_turn, struct BoardState *board_state, unsigned int **piece_count, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status);

/* This function resets the chosen coordinates and pawn en passant status for the new turn */
void ResetCoordinatesAndPawnEnPassantNewTurn(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status);

/* This function prints the current values of a BoardState */
void PrintBoardState(struct BoardState *board_state);

/* This function prints the evaluation score based on position and game phase */
void PrintPositionGamePhaseScore(struct BoardState *board_state);

/******************************************************************************************************/
/******************************************** HUMAN TURN **********************************************/
/******************************************************************************************************/

/* This function executes a human turn */
int HumanTurn(struct PlayerOptions *player_options, struct BoardState *board_state, char *basic_chessboard, int **square, int **pawn_en_passant_status, unsigned int **piece_count, char *read_coordinates, int *piece_coordinates, int *move_coordinates, int ****piece_positions, int **color, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history);

/* This function selects piece coordinates */
int SelectPieceCoordinates(struct BoardState *board_state, char *read_coordinates, int *piece_coordinates, int **square, int ****piece_positions);

/* This function converts column coordinates from char to int */
void ConvertReadColumnCoordinatesChartoInt(char *read_coordinates, int *variable_coordinates);

/* This function converts row coordinates from char to int */
void ConvertReadRowCoordinatesChartoInt(char *read_coordinates, int *variable_coordinates);

/* This function prints piece selected */
void PrintPieceSelected(int **square, int *piece_coordinates);

/* This function selects move coordinates */
int SelectMoveCoordinates(struct BoardState *board_state, char *read_coordinates, int *move_coordinates, int *piece_coordinates, int **square);

/* This function checks the validity of human move */
int CheckHumanMoveValidity(struct BoardState *board_state, int **square, char *read_coordinates, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int ****piece_positions, int *skip);

/******************************************************************************************************/
/*************************************** MOVEMENT VALIDATION ******************************************/
/******************************************************************************************************/

/* This function validates movements */
int MovementValidation(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int ****piece_positions, int human);

/* This function checks if white pawn can move */
void WhitePawnMovement(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int *invalid_move, int mate_checking);

/* This function checks if black pawn can move */
void BlackPawnMovement(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int *invalid_move, int mate_checking);

/* This function checks if knight can move */
void KnightMovement(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move);

/* This function checks if bishop can move */
void BishopMovement(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move);

/* This function checks if rook can move */
void RookMovement(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move);

/* This function checks if queen can move */
void QueenMovement(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move);

/* This function checks if king can move */
void KingMovement(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move);

/******************************************************************************************************/
/**************************************** SPECIAL MOVEMENTS *******************************************/
/******************************************************************************************************/

/* This function checks if a white pawn is en passant */
void WhiteEnPassant(int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int human);

/* This function checks if a black pawn is en passant */
void BlackEnPassant(int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int human);

/* This function checks if white queen side castling is valid */
void WhiteQueenSideCastling(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int *invalid_move);

/* This function checks if black queen side castling is valid */
void BlackQueenSideCastling(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int *invalid_move);

/* This function checks if white queen king castling is valid */
void WhiteKingSideCastling(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int *invalid_move);

/* This function checks if black king side castling is valid */
void BlackKingSideCastling(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int *invalid_move);

/* This function promotes pawns that reach other side of board */
void PawnPromotion(int **square, int *move_coordinates);

/* This function disallows castling */
void DisallowCastling(struct BoardState *board_state, int **square);

/******************************************************************************************************/
/***************************************** CHECK FOR CHECK ********************************************/
/******************************************************************************************************/

/* This function checks valid moves for check */
int CheckValidMovesForCheck(int *piece_coordinates, int *move_coordinates, int **square, int ****piece_positions);

/* This function checks if in check */
int InCheckChecker(int **square, int *king_coordinates, int check_color);

/* This function checks if white is in check due to pawns */
int WhiteInCheckDueToPawns(int **square, int *king_coordinates);

/* This function checks if black is in check due to pawns */
int BlackInCheckDueToPawns(int **square, int *king_coordinates);

/* This function checks if in check due to knights */
int InCheckDueToKnights(int **square, int *king_coordinates, int check_color);

/* This function checks if in check due to horizontals */
int InCheckDueToHorizontals(int **square, int *king_coordinates, int check_color);

/* This function checks if in check due to verticals */
int InCheckDueToVerticals(int **square, int *king_coordinates, int check_color);

/* This function checks if in check due to diagonals */
int InCheckDueToDiagonals(int **square, int *king_coordinates, int check_color);

/******************************************************************************************************/
/****************************************** PERFORM MOVES *********************************************/
/******************************************************************************************************/

/* This function performs the selected human moves */
int PerformHumanMoves(struct PlayerOptions *player_options, struct BoardState *board_state, char *basic_chessboard, int turn, int **square, int **pawn_en_passant_status, unsigned int **piece_count, int *piece_coordinates, int *move_coordinates, int ****piece_positions, int **color, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history);

/* This function performs validated moves */
void PerformValidatedMoves(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int ****piece_positions, int human);

/******************************************************************************************************/
/*************************************** CHECK FOR GAME OVER ******************************************/
/******************************************************************************************************/

/* This function checks if the game is over according to the rules in its current state */
int CheckIfGameOver(struct BoardState *board_state, int turn, int **square, int **pawn_en_passant_status, unsigned int **piece_count, int *move_coordinates, int ****piece_positions, int **color, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int human);

/* This function checks for checkmate and stalemate */
int NoLegalMovesMateChecker(struct BoardState *board_state, int turn, int **square, int **pawn_en_passant_status, unsigned int **piece_count, int ****piece_positions, int **color, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int human);

/* This function checks move validity for check */
void CheckHumanMoveValidityForCheck(int *piece_coordinates, int *move_coordinates, int **square, int ****piece_positions, int *invalid_move, unsigned int *moves_valid);

/******************************************************************************************************/
/******************************************** TRY MOVES ***********************************************/
/******************************************************************************************************/

/* This function tries all white pawn moves */
void TryWhitePawnMoves(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state);

/* This function applies white pawn moves */
void ApplyWhitePawnMoves(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int *invalid_move, int mate_checking);

/* This function tries all black pawn moves */
void TryBlackPawnMoves(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state);

/* This function applies black pawn moves */
void ApplyBlackPawnMoves(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int *invalid_move, int mate_checking);

/* This function tries all knight moves */
void TryKnightMoves(int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state);

/* This function applies knight moves */
void ApplyKnightMoves(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move);

/* This function tries all bishop moves */
void TryBishopMoves(int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state);

/* This function applies bishop moves */
void ApplyBishopMoves(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move);

/* This function tries all rook moves */
void TryRookMoves(int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state);

/* This function applies rook moves */
void ApplyRookMoves(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move);

/* This function tries all queen moves */
void TryQueenMoves(int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state);

/* This function tries all king moves */
void TryKingMoves(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state);

/* This function applies king moves */
void ApplyKingMoves(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move);

/******************************************************************************************************/
/********************************** SPECIAL GAME OVER CONDITIONS **************************************/
/******************************************************************************************************/

/* This function checks for three fold repetition */
int ThreeFoldRepetition(int turn, int **square, int ***board_history);

/* This function checks if checmate is impossible */
int CheckmateImpossibility(unsigned int **piece_count, int **square, int **color);

/* This function checks the fifty move rule of no pawn move AND piece capture in fifty moves */
int FiftyMoveRule(struct BoardState *board_state, int *move_coordinates, int **square);

/******************************************************************************************************/
/************************************* BOARD EVALUATION SCORE *****************************************/
/******************************************************************************************************/

/* This function evaluates the boards score */
int BoardEvaluationScore(struct BoardState *board_state, int turn, int **square, int **color, int ****piece_positions, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers);

/******************************************************************************************************/
/****************************************** COMPUTER TURN *********************************************/
/******************************************************************************************************/

/* This function is a computer player playing white */
int WhiteComputer(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct PlayerOptions *player_options, struct BoardState *board_state, int max_depth, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int **square, int **color, int ****piece_positions, unsigned int **piece_count, int *piece_value, int **pawn_en_passant_status, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int *piece_coordinates, int *move_coordinates, int ***board_history, char *basic_chessboard);

/* This function is a computer player playing white */
int BlackComputer(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct PlayerOptions *player_options, struct BoardState *board_state, int max_depth, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int **square, int **color, int ****piece_positions, unsigned int **piece_count, int *piece_value, int **pawn_en_passant_status, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int *piece_coordinates, int *move_coordinates, int ***board_history, char *basic_chessboard);

/* This function performs computer moves */
int ComputerMoves(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int **color, int ****piece_positions, unsigned int **piece_count, int *piece_value, int **pawn_en_passant_status, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int *piece_coordinates, int *move_coordinates, int ***board_history);

/* This function performs Minimax algorithm */
int Minimax(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int **color, int ****piece_positions, unsigned int **piece_count, int *piece_value, int **pawn_en_passant_status, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int *piece_coordinates, int *move_coordinates, int ***board_history, int *alpha, int *beta);

/* This function applies minimax to white pawns */
void MinimaxWhitePawn(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta);

/* This function applies minimax to black pawns */
void MinimaxBlackPawn(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta);

/* This function applies minimax to knights */
void MinimaxKnight(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta);

/* This function applies minimax to bishops */
void MinimaxBishop(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta);

/* This function applies minimax to rooks */
void MinimaxRook(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta);

/* This function applies minimax to queens */
void MinimaxQueen(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta);

/* This function applies minimax to kings */
void MinimaxKing(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta);

/* This function applies minimax unoptimized */
void MinimaxUnoptimized(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta);

/* This function attempts computer moves */
void AttemptComputerMoves(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int move_file, int move_rank, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta);

/* This function checks if computer move is legal */
int CheckLegalComputerMove(struct BoardState *board_state, int turn, int depth, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int ****piece_positions);

/* This function applies computer moves */
void ApplyComputerMove(struct BoardState *board_state, int depth, int piece_file, int piece_rank, int move_file, int move_rank, int *old_move_square, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int ****piece_positions);

/* This function performs remaining operations after applying computer moves */
int AfterApplyingComputerMove(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int depth, int max_depth, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int move_file, int move_rank, int old_move_square, int **square, int **color, int *piece_coordinates, int *move_coordinates, unsigned int **piece_count, int ****piece_positions, int **pawn_en_passant_status, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int promotion, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta);

/* This function undos computer moves */
void UndoComputerMove(struct BoardState *board_state, int depth, int **square, int *piece_coordinates, int *move_coordinates, int old_piece_square, int old_move_square, int **pawn_en_passant_status, struct BoardState old_board_state);

/* This function tries computer white pawn moves */
int ComputerWhitePawnMoves(int *piece_coordinates, int *move_coordinates, int move_number);

/* This function tries computer black pawn moves */
int ComputerBlackPawnMoves(int *piece_coordinates, int *move_coordinates, int move_number);

/* This function tries computer knight moves */
int ComputerKnightMoves(int *piece_coordinates, int *move_coordinates, int move_number);

/* This function tries computer king moves */
int ComputerKingMoves(int turn, int depth, int *piece_coordinates, int *move_coordinates, int move_number);

/******************************************************************************************************/
/************************************************ MAIN ************************************************/
/******************************************************************************************************/

int main(int argc, char *argv[])
{
	unsigned int i, j, k;
	int systemreturn, invalid_move = 1, skip = 0;
	char read_coordinates[2];

	/******************************************************************************************************/
	/******************************************** BOARD SETUP *********************************************/
	/******************************************************************************************************/

	/* Initialize chess constants */
	InitializeChessConstants();

	/* Create structure to track all player chosen options */
	struct PlayerOptions player_options;
	InitializePlayerOptions(&player_options);

	/* Create current board state structure to track all state flags */
	struct BoardState board_state;

	char basic_chessboard[5588]; // this holds the basic plot script code to write out the custom plot scripts

	int **square; // what piece occupies each square 0=Empty, 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King	Positive = White, Negative = Black	[0][0] in bottom left corner

	int **color; // the board's color for each square 0 = white, 1 = black

	BoardSetup(&board_state, basic_chessboard, &square, &color);

	/******************************************************************************************************/
	/******************************************** PIECE SETUP *********************************************/
	/******************************************************************************************************/

	unsigned int **piece_count; // first dimension = color, second dimension = piece type

	int ****piece_positions; // the positions of each piece. Color, piece type, piece number, coordinate

	int **pawn_en_passant_status; // the status of enpassant for each pawn 0 = white, 1 = black

	PieceSetup(&board_state, square, &piece_count, &piece_positions, &pawn_en_passant_status);

	/******************************************************************************************************/
	/******************************************** PIECE VALUES ********************************************/
	/******************************************************************************************************/

	int *piece_value; // the relative value of pieces

	int **non_passed_pawn_opening_values; // the values of non-passed pawns in the opening

	int **non_passed_pawn_end_game_values; // the values of non-passed pawns in the end_game

	double **pawn_advancement_multipliers; // the multipliers for pawn advancement

	double **knight_position_modifiers; // modifies the value of knights based on their positions

	InitializePieceAndPositionValues(&piece_value, &non_passed_pawn_opening_values, &non_passed_pawn_end_game_values, &pawn_advancement_multipliers, &knight_position_modifiers);

	/******************************************************************************************************/
	/********************************************* GAME SETUP *********************************************/
	/******************************************************************************************************/

	int *piece_coordinates; // the column and row coordinates of selected piece location

	int *move_coordinates; // the column and row coordinates of move location

	int ***board_history; // the board_history of each board layout

	GameSetup(player_options.board_history_length, square, &piece_coordinates, &move_coordinates, &board_history);

	FILE *outfile_turn;
	outfile_turn = fopen("turn.txt", "w");
	fprintf(outfile_turn, "%d\n", board_state.turn);
	fclose(outfile_turn);

	/******************************************************************************************************/
	/************************************************ PLAY ************************************************/
	/******************************************************************************************************/

	/******************************************************************************************************/
	/******************************************* INITIAL BOARD ********************************************/
	/******************************************************************************************************/

	PrintPieces(board_state.turn, square, color, player_options.players, basic_chessboard);

	/******************************************************************************************************/
	/******************************************* BEGIN PLAYING ********************************************/
	/******************************************************************************************************/

	GetNumberOfPlayers(&player_options);

	if (player_options.players == 2) // if you want to play against another person
	{
		/******************************************************************************************************/
		/********************************************* TWO PLAYER *********************************************/
		/******************************************************************************************************/
		printf("You are playing two player!\n");

		board_state.evaluation_score = BoardEvaluationScore(&board_state, board_state.turn, square, color, piece_positions, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers);

		PrintPositionGamePhaseScore(&board_state);

		while (skip <= 1)
		{
			StartNewTurn(outfile_turn, &board_state, piece_count, piece_coordinates, move_coordinates, pawn_en_passant_status);

			skip = HumanTurn(&player_options, &board_state, basic_chessboard, square, pawn_en_passant_status, piece_count, read_coordinates, piece_coordinates, move_coordinates, piece_positions, color, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history);
		} // end of while loop
	}
	else // if you want at least one computer
	{
		/******************************************************************************************************/
		/****************************************** ADD COMPUTER(S) *******************************************/
		/******************************************************************************************************/

		FILE *outfile_computer_move_log = fopen("outputs/computermovelog.txt", "w"); // write only
		FILE *outfile_computer_move_score_log = fopen("outputs/ComputerMovescorelog.txt", "w"); // write only
		FILE *outfile_piece_move_board_history = fopen("outputs/piecemove_board_history.txt", "w"); // write only

		int *depth_daughters = malloc(sizeof(int) * maximum_depth);
		int *depth_valid_move_count = malloc(sizeof(int) * maximum_depth);

		int **piece_board_history;
		int **move_board_history;

		CreateComputerOnlyLoggingVariables(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, maximum_depth, depth_daughters, depth_valid_move_count, &piece_board_history, &move_board_history);

		if (player_options.players == 1) // if you want to play against computer
		{
			/******************************************************************************************************/
			/******************************************* SINGLE PLAYER ********************************************/
			/******************************************************************************************************/

			printf("You are playing against computer!\n");

			GetComputerDifficulty(maximum_depth, 1, &player_options.difficulty_computer1);

			GetSinglePlayerColorChoice(&player_options);

			while (skip <= 1)
			{
				/******************************************************************************************************/
				/********************************************* NEW TURN ***********************************************/
				/******************************************************************************************************/

				StartNewTurn(outfile_turn, &board_state, piece_count, piece_coordinates, move_coordinates, pawn_en_passant_status);

				for (i = 0; i < maximum_depth; i++)
				{
					depth_daughters[i] = 0;
					depth_valid_move_count[i] = 0;
				} // end of i loop

				if (board_state.turn % 2 == 1) // if it is white's turn
				{
					if (player_options.color_choice == 0) // if player is white
					{
						/******************************************************************************************************/
						/******************************************** WHITE HUMAN *********************************************/
						/******************************************************************************************************/

						skip = HumanTurn(&player_options, &board_state, basic_chessboard, square, pawn_en_passant_status, piece_count, read_coordinates, piece_coordinates, move_coordinates, piece_positions, color, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history);
					} // end of if player is white
					else // if player is black
					{
						/******************************************************************************************************/
						/******************************************* BLACK COMPUTER *******************************************/
						/******************************************************************************************************/

						if (BlackComputer(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, &player_options, &board_state, player_options.difficulty_computer1, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, square, color, piece_positions, piece_count, piece_value, pawn_en_passant_status, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, piece_coordinates, move_coordinates, board_history, basic_chessboard) == 1)
						{
							break; // break while loop since game is over
						}

					} // end of if player is black
				} // end of if it is white's turn
				else // if it is black's turn
				{
					if (player_options.color_choice == 0) // if player is white
					{
						/******************************************************************************************************/
						/******************************************* WHITE COMPUTER *******************************************/
						/******************************************************************************************************/

						if (WhiteComputer(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, &player_options, &board_state, player_options.difficulty_computer1, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, square, color, piece_positions, piece_count, piece_value, pawn_en_passant_status, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, piece_coordinates, move_coordinates, board_history, basic_chessboard) == 1)
						{
							break; // break while loop since game is over
						}
					} // end of if player is white
					else // if player is black
					{
						/******************************************************************************************************/
						/******************************************** BLACK HUMAN *********************************************/
						/******************************************************************************************************/

						skip = HumanTurn(&player_options, &board_state, basic_chessboard, square, pawn_en_passant_status, piece_count, read_coordinates, piece_coordinates, move_coordinates, piece_positions, color, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history);
					} // end of if player is black
				} // end of if it is black's turn
			} // end of while loop
		} // end of if you want to play against computer
		else // if you want to simulate
		{
			/******************************************************************************************************/
			/********************************************* SIMULATION *********************************************/
			/******************************************************************************************************/

			printf("You are simulating!\n");

			GetComputerDifficulty(maximum_depth, 1, &player_options.difficulty_computer1);

			GetComputerDifficulty(maximum_depth, 2, &player_options.difficulty_computer2);

			while (board_state.game_over == 0)
			{
				StartNewTurn(outfile_turn, &board_state, piece_count, piece_coordinates, move_coordinates, pawn_en_passant_status);

				for (i = 0; i < maximum_depth; i++)
				{
					depth_daughters[i] = 0;
					depth_valid_move_count[i] = 0;
				} // end of i loop

				if (board_state.turn % 2 == 1) // if it is white's turn
				{
					/******************************************************************************************************/
					/******************************************* WHITE COMPUTER *******************************************/
					/******************************************************************************************************/

					if (WhiteComputer(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, &player_options, &board_state, player_options.difficulty_computer1, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, square, color, piece_positions, piece_count, piece_value, pawn_en_passant_status, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, piece_coordinates, move_coordinates, board_history, basic_chessboard) == 1)
					{
						break; // break while loop since game is over
					}
				} // end of if it is white's turn
				else // if it is black's turn
				{
					/******************************************************************************************************/
					/********************************************* BLACK COMPUTER *********************************************/
					/******************************************************************************************************/

					if (BlackComputer(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, &player_options, &board_state, player_options.difficulty_computer2, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, square, color, piece_positions, piece_count, piece_value, pawn_en_passant_status, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, piece_coordinates, move_coordinates, board_history, basic_chessboard) == 1)
					{
						break; // break while loop since game is over
					}
				} // end of if it is black's turn

				if (board_state.turn == moves_to_break_computer_loop)
				{
					break; // break while loop since game is over
				}
			} // end of while
		} // end of if you want to simulate

		/******************************************************************************************************/
		/************************************ FREE COMPUTER DYNAMIC MEMORY ************************************/
		/******************************************************************************************************/

		for (i = 0; i < maximum_depth; i++)
		{
			free(move_board_history[i]);
			free(piece_board_history[i]);
		} // end of i loop
		free(move_board_history);
		free(piece_board_history);

		free(depth_valid_move_count);
		free(depth_daughters);

		fclose(outfile_piece_move_board_history);
		fclose(outfile_computer_move_score_log);
		fclose(outfile_computer_move_log);
	} // end of if you want at least one computer

	/******************************************************************************************************/
	/**************************************** FREE DYNAMIC MEMORY *****************************************/
	/******************************************************************************************************/

	/* Free game setup arrays */
	free(move_coordinates);
	free(piece_coordinates);
	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		for (j = 0; j < chess_constants.number_of_rows; j++)
		{
			free(board_history[i][j]);
		} // end of j loop
		free(board_history[i]);
	} // end of i loop
	free(board_history);

	/* Free piece value arrays */
	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		free(knight_position_modifiers[i]);
		free(non_passed_pawn_end_game_values[i]);
		free(non_passed_pawn_opening_values[i]);
	} // end of i loop
	free(knight_position_modifiers);
	free(non_passed_pawn_end_game_values);
	free(non_passed_pawn_opening_values);

	for (i = 0; i < 4; i++)
	{
		free(pawn_advancement_multipliers[i]);
	} // end of i loop
	free(pawn_advancement_multipliers);
	free(piece_value);

	/* Free piece setup arrays */
	for (i = 0; i < chess_constants.number_of_colors; i++)
	{
		for (j = 0; j < chess_constants.number_of_unique_pieces; j++)
		{
			for (k = 0; k < chess_constants.max_pieces_per_type[j]; k++)
			{
				free(piece_positions[i][j][k]);
			} // end of k loop
			free(piece_positions[i][j]);
		} // end of j loop
		free(pawn_en_passant_status[i]);
		free(piece_positions[i]);
		free(piece_count[i]);
	} // end of i loop
	free(pawn_en_passant_status);
	free(piece_positions);
	free(piece_count);

	/* Free board setup arrays */
	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		free(color[i]);
		free(square[i]);
	} // end of i loop
	free(color);
	free(square);
} // end of main

/******************************************************************************************************/
/********************************************* FUNCTIONS **********************************************/
/******************************************************************************************************/

/******************************************************************************************************/
/******************************************** BOARD SETUP *********************************************/
/******************************************************************************************************/

/* This function initializes chess constants */
void InitializeChessConstants()
{
	chess_constants.number_of_colors = 2;
	chess_constants.number_of_rows = 8;
	chess_constants.number_of_columns = 8;
	chess_constants.number_of_unique_pieces = 6;

	chess_constants.max_pieces_per_type[0] = 8;
	chess_constants.max_pieces_per_type[1] = 2 + chess_constants.max_pieces_per_type[0];
	chess_constants.max_pieces_per_type[2] = 2 + chess_constants.max_pieces_per_type[0];
	chess_constants.max_pieces_per_type[3] = 2 + chess_constants.max_pieces_per_type[0];
	chess_constants.max_pieces_per_type[4] = 1 + chess_constants.max_pieces_per_type[0];
	chess_constants.max_pieces_per_type[5] = 1;

	chess_constants.number_of_possible_pawn_moves = 4;
	chess_constants.number_of_possible_knight_moves = 8;
	chess_constants.number_of_possible_king_moves = 10;
} // end of InitializeChessConstants function

/* This function initializes the player chosen options */
void InitializePlayerOptions(struct PlayerOptions *player_options)
{
	player_options->board_history_length = 6000;

	player_options->players = -9;
	player_options->color_choice = -9;

	player_options->difficulty_computer1 = -9;
	player_options->difficulty_computer2 = -9;
} // end of InitializePlayerOptions function

/* This function sets up the board for play */
void BoardSetup(struct BoardState *board_state, char *basic_chessboard, int ***square, int ***color)
{
	unsigned int i;
	int systemreturn;

	InitializeBoardStateVariables(board_state);

	FILE *infile_basic_chessboard = fopen("plotscripts/basic_chessboard.txt", "r"); // read only
	for (i = 0; i < 5588; i++)
	{
		systemreturn = fscanf(infile_basic_chessboard, "%c", &basic_chessboard[i]);
		if (systemreturn == -1)
		{
			printf("Failed reading basic_chessboard.txt\n");
		}
	}
	fclose(infile_basic_chessboard);

	(*square) = malloc(sizeof(int*) * chess_constants.number_of_columns);
	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		(*square)[i] = malloc(sizeof(int) * chess_constants.number_of_rows);
	} // end of i loop

	(*color) = malloc(sizeof(int*) * chess_constants.number_of_columns);
	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		(*color)[i] = malloc(sizeof(int) * chess_constants.number_of_rows);
	} // end of i loop

	if (start_with_partial_board == 0) // if we are starting with a normal board
	{
		InitalBoardSetup((*square), (*color));
	} // end of if we are starting with a normal board
	else // if we are starting with a partial board
	{
		PartialBoardSetup((*square), (*color));
	} // end of if we are starting with a partial board
} // end of BoardSetup function

/* This function initializes the board state variables */
void InitializeBoardStateVariables(struct BoardState *board_state)
{
	board_state->turn = 0;

	board_state->white_pieces = 0;
	board_state->black_pieces = 0;
	board_state->white_pieces_old = 0;
	board_state->black_pieces_old = 0;

	board_state->white_pieces = 0;
	board_state->black_pieces = 0;
	board_state->white_pieces_old = 0;
	board_state->black_pieces_old = 0;

	board_state->white_in_check = 0;
	board_state->black_in_check = 0;

	board_state->white_queen_side_castle = 1;
	board_state->black_queen_side_castle = 1;
	board_state->white_king_side_castle = 1;
	board_state->black_king_side_castle = 1;

	board_state->white_queen_side_castle_allow = 0;
	board_state->black_queen_side_castle_allow = 0;
	board_state->white_king_side_castle_allow = 0;
	board_state->black_king_side_castle_allow = 0;

	board_state->en_passant_captured = -9;

	board_state->pawn_move = 0;
	board_state->capture_move = 0;

	board_state->evaluation_score = 0;

	board_state->closed_position = 0;
	board_state->end_game = 0;

	board_state->game_over = 0;
} // end of InitializeBoardStateVariables function

/* This function setups the initial board */
void InitalBoardSetup(int **square, int **color)
{
	int i, j;

	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		for (j = 0; j < chess_constants.number_of_rows; j++)
		{
			if (j == 1) // if second from from the bottom
			{
				square[i][j] = 1; // white pawns
			} // end of if second from from the bottom
			else if (j == 6) // if second from from the top
			{
				square[i][j] = -1; // black pawns
			} // end of if second from from the top
			else // if not a pawn row
			{
				square[i][j] = 0; // rest of the squares empty
			} // end of if not a pawn row

			if (i%2 == 1) // if an odd row
			{
				if (j%2 == 1) // if an odd column
				{
					color[i][j] = 1; // black
				} // end of if an odd column
				else // if an even column
				{
					color[i][j] = 0; // white
				} // end of if an even column
			} // end of if an odd row
			else // if an even row
			{
				if (j%2 == 1) // if an odd column
				{
					color[i][j] = 0; // white
				} // end of if an odd column
				else // if an even column
				{
					color[i][j] = 1; // black
				} // end of if an even column
			} // end of if an even row
		} // end of j loop
	} // end of i loop

	square[0][0] = 4; // white rook
	square[7][0] = 4; // white rook

	square[0][7] = -4; // black rook
	square[7][7] = -4; // black rook

	square[1][0] = 2; // white knight
	square[6][0] = 2; // white knight

	square[1][7] = -2; // black knight
	square[6][7] = -2; // black knight

	square[2][0] = 3; // white bishop
	square[5][0] = 3; // white bishop

	square[2][7] = -3; // black bishop
	square[5][7] = -3; // black bishop

	square[3][0] = 5; // white queen
	square[4][0] = 6; // white king

	square[3][7] = -5; // black queen
	square[4][7] = -6; // black king
} // end of InitalBoardSetup function

/* This function setups a partial board */
void PartialBoardSetup(int **square, int **color)
{
	int i, j;

	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		for (j = 0; j < chess_constants.number_of_rows; j++)
		{
			square[i][j] = 0; // rest of the squares empty

			if (i%2 == 1) // if an odd row
			{
				if (j%2 == 1) // if an odd column
				{
					color[i][j] = 1; // black
				} // end of if an odd column
				else // if an even column
				{
					color[i][j] = 0; // white
				} // end of if an even column
			} // end of if an odd row
			else // if an even row
			{
				if (j%2 == 1) // if an odd column
				{
					color[i][j] = 0; // white
				} // end of if an odd column
				else // if an even column
				{
					color[i][j] = 1; // black
				} // end of if an even column
			} // end of if an even row
		} // end of j loop
	} // end of i loop

	/* Set these as desired to pick up from an already started game */
	square[0][2] = 1; // white pawn
//	square[3][1] = 1; // white pawn

//	square[4][6] = 5; // white queen

	square[0][0] = 6; // white king

//	square[3][5] = -1; // black pawn
//	square[3][6] = -1; // black pawn

	square[7][7] = -6; // black king
} // end of PartialBoardSetup function

/******************************************************************************************************/
/******************************************** PIECE SETUP *********************************************/
/******************************************************************************************************/

/* This function sets up piece counts and position and pawn en passant status */
void PieceSetup(struct BoardState *board_state, int **square, unsigned int ***piece_count, int *****piece_positions, int ***pawn_en_passant_status)
{
	unsigned int i, j, k, l;

	(*piece_count) = malloc(sizeof(unsigned int*) * chess_constants.number_of_colors);
	for (i = 0; i < chess_constants.number_of_colors; i++)
	{
		(*piece_count)[i] = malloc(sizeof(unsigned int) * chess_constants.number_of_unique_pieces);
	} // end of i loop

	(*piece_positions) = malloc(sizeof(int***) * chess_constants.number_of_colors);
	for (i = 0; i < chess_constants.number_of_colors; i++)
	{
		(*piece_positions)[i] = malloc(sizeof(int**) * chess_constants.number_of_unique_pieces);
		for (j = 0; j < chess_constants.number_of_unique_pieces; j++)
		{
			(*piece_positions)[i][j] = malloc(sizeof(int*) * chess_constants.max_pieces_per_type[j]);
			for (k = 0; k < chess_constants.max_pieces_per_type[j]; k++)
			{
				(*piece_positions)[i][j][k] = malloc(sizeof(int) * 2);
				for (l = 0; l < 2; l++)
				{
					(*piece_positions)[i][j][k][l] = -9;
				} // end of l loop
			} // end of k loop
		} // end of j loop
	} // end of i loop

	(*pawn_en_passant_status) = malloc(sizeof(int*) * chess_constants.number_of_colors);
	for (i = 0; i < chess_constants.number_of_colors; i++)
	{
		(*pawn_en_passant_status)[i] = malloc(sizeof(int) * chess_constants.number_of_columns);
		for (j = 0; j < chess_constants.number_of_columns; j++)
		{
			(*pawn_en_passant_status)[i][j] = 0;
		} // end of j loop
	} // end of i loop

	CountPieces(board_state, square, (*piece_count), (*piece_positions));

	board_state->white_pieces_old = board_state->white_pieces;
	board_state->black_pieces_old = board_state->black_pieces;
} // end of PieceSetup function

/* This function counts the number of pieces */
void CountPieces(struct BoardState *board_state, int **square, unsigned int **piece_count, int ****piece_positions)
{
	int i, j;
	unsigned color_index, piece_index;

	board_state->white_pieces = 0;
	board_state->black_pieces = 0;

	for (i = 0; i < chess_constants.number_of_unique_pieces; i++)
	{
		piece_count[0][i] = 0;
		piece_count[1][i] = 0;
	} // end of i loop

	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		for (j = 0; j < chess_constants.number_of_rows; j++)
		{
			if (square[i][j] != 0)
			{
				color_index = square[i][j] > 0 ? 0 : 1;
				piece_index = square[i][j] > 0 ? square[i][j] - 1 : -square[i][j] - 1;

				piece_positions[color_index][piece_index][piece_count[color_index][piece_index]][0] = i;
				piece_positions[color_index][piece_index][piece_count[color_index][piece_index]][1] = j;

				piece_count[color_index][piece_index]++;

				if (color_index == 0) // if square has white piece
				{
					board_state->white_pieces++;
				} // end of if square has white piece
				else if (color_index == 1) // if square has black piece
				{
					board_state->black_pieces++;
				} // end of if square has black piece
			}
		} // end of j loop
	} // end of i loop
} // end of CountPieces function

/* This function prints the pieces onto the board */
void PrintPieces(int turn, int **square, int **color, int players, char *basic_chessboard)
{
	unsigned int i, j;
	int systemreturn;

	FILE *outfile_chessboard_gplot = fopen("plotscripts/chessboard_artwork.gplot", "w"); // write only
	FILE *outfile_chessboard_simulation_gplot = fopen("plotscripts/chessboard_simulation_artwork.gplot", "w"); // write only

	fprintf(outfile_chessboard_gplot, "set term jpeg size 1920, 1200\n");
	fprintf(outfile_chessboard_gplot, "set output \"chessboard.jpeg\"\n");

	fprintf(outfile_chessboard_simulation_gplot, "t = system(\"cat turn.txt\")\n");
	fprintf(outfile_chessboard_simulation_gplot, "set term jpeg size 1920, 1200\n");
	fprintf(outfile_chessboard_simulation_gplot, "set output \"images/chessboard_\".t.\".jpeg\"\n");

	for (i = 0; i < 5588; i++)
	{
		fprintf(outfile_chessboard_gplot, "%c", basic_chessboard[i]);
		fprintf(outfile_chessboard_simulation_gplot, "%c", basic_chessboard[i]);
	}
	fprintf(outfile_chessboard_gplot, "\nplot ");
	fprintf(outfile_chessboard_simulation_gplot, "\nplot ");

	FILE *outfile_black_pieces_black_squares = fopen("plotscripts/black_pieces, blacksquares.txt", "w"); // write only
	FILE *outfile_black_pieces_white_squares = fopen("plotscripts/black_pieces, whitesquares.txt", "w"); // write only
	FILE *outfile_white_pieces_black_squares = fopen("plotscripts/white_pieces, blacksquares.txt", "w"); // write only
	FILE *outfile_white_pieces_white_squares = fopen("plotscripts/white_pieces, whitesquares.txt", "w"); // write only

	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		for (j = 0; j < chess_constants.number_of_rows; j++)
		{
			if (color[i][j] == 1) // if it's a black square
			{
				if (square[i][j] == -1) // if it's a black pawn
				{
					fprintf(outfile_black_pieces_black_squares, "%lf\t%lf\tP\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_black_pawn_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_black_pawn_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == -2) // if it's a black knight
				{
					fprintf(outfile_black_pieces_black_squares, "%lf\t%lf\tN\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, " \"plotscripts/chess_piece_black_knight_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, " \"plotscripts/chess_piece_black_knight_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == -3) // if it's a black bishop
				{
					fprintf(outfile_black_pieces_black_squares, "%lf\t%lf\tB\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_black_bishop_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_black_bishop_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == -4) // if it's a black rook
				{
					fprintf(outfile_black_pieces_black_squares, "%lf\t%lf\tR\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_black_rook_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_black_rook_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == -5) // if it's a black queen
				{
					fprintf(outfile_black_pieces_black_squares, "%lf\t%lf\tQ\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_black_queen_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_black_queen_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == -6) // if it's a black King
				{
					fprintf(outfile_black_pieces_black_squares, "%lf\t%lf\tK\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_black_king_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_black_king_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 1) // if it's a white pawn
				{
					fprintf(outfile_white_pieces_black_squares, "%lf\t%lf\tP\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_pawn_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_pawn_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 2) // if it's a white knight
				{
					fprintf(outfile_white_pieces_black_squares, "%lf\t%lf\tN\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_knight_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_knight_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 3) // if it's a white bishop
				{
					fprintf(outfile_white_pieces_black_squares, "%lf\t%lf\tB\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_bishop_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_bishop_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 4) // if it's a white rook
				{
					fprintf(outfile_white_pieces_black_squares, "%lf\t%lf\tR\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_rook_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_rook_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 5) // if it's a white queen
				{
					fprintf(outfile_white_pieces_black_squares, "%lf\t%lf\tQ\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_queen_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_queen_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 6) // if it's a white King
				{
					fprintf(outfile_white_pieces_black_squares, "%lf\t%lf\tK\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_king_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_king_on_black.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
			} // end of if it's a black square
			else // if it's a white square
			{
				if (square[i][j] == -1) // if it's a black pawn
				{
					fprintf(outfile_black_pieces_white_squares, "%lf\t%lf\tP\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_black_pawn.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_black_pawn.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == -2) // if it's a black knight
				{
					fprintf(outfile_black_pieces_white_squares, "%lf\t%lf\tN\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, " \"plotscripts/chess_piece_black_knight.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, " \"plotscripts/chess_piece_black_knight.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == -3) // if it's a black bishop
				{
					fprintf(outfile_black_pieces_white_squares, "%lf\t%lf\tB\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_black_bishop.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_black_bishop.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == -4) // if it's a black rook
				{
					fprintf(outfile_black_pieces_white_squares, "%lf\t%lf\tR\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_black_rook.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_black_rook.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == -5) // if it's a black queen
				{
					fprintf(outfile_black_pieces_white_squares, "%lf\t%lf\tQ\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_black_queen.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_black_queen.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == -6) // if it's a black King
				{
					fprintf(outfile_black_pieces_white_squares, "%lf\t%lf\tK\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_black_king.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_black_king.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 1) // if it's a white pawn
				{
					fprintf(outfile_white_pieces_white_squares, "%lf\t%lf\tP\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_pawn.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_pawn.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 2) // if it's a white knight
				{
					fprintf(outfile_white_pieces_white_squares, "%lf\t%lf\tN\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_knight.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_knight.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 3) // if it's a white bishop
				{
					fprintf(outfile_white_pieces_white_squares, "%lf\t%lf\tB\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_bishop.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_bishop.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 4) // if it's a white rook
				{
					fprintf(outfile_white_pieces_white_squares, "%lf\t%lf\tR\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_rook.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_rook.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 5) // if it's a white queen
				{
					fprintf(outfile_white_pieces_white_squares, "%lf\t%lf\tQ\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_queen.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_queen.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
				else if (square[i][j] == 6) // if it's a white King
				{
					fprintf(outfile_white_pieces_white_squares, "%lf\t%lf\tK\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_gplot, "\"plotscripts/chess_piece_white_king.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
					fprintf(outfile_chessboard_simulation_gplot, "\"plotscripts/chess_piece_white_king.png\" binary filetype = png center = (%lf,%lf) dx = 0.01 dy = 0.01 with rgbimage notitle,	\\\n", i + 0.5, j + 0.5);
				}
			} // end of if it's a white square
		} // end of j loop
	} // end of i loop

	fclose(outfile_black_pieces_black_squares);
	fclose(outfile_black_pieces_white_squares);
	fclose(outfile_white_pieces_black_squares);
	fclose(outfile_white_pieces_white_squares);

	fclose(outfile_chessboard_simulation_gplot);
	fclose(outfile_chessboard_gplot);

	if (turn > 0)
	{
		if (players > 0)
		{
			if (use_actual_piece_artwork == 1)
			{
				systemreturn = system("gnuplot plotscripts/chessboard_artwork.gplot");
			}
			else
			{
				systemreturn = system("gnuplot plotscripts/chessboard_basic.gplot");
			}

			if (systemreturn == -1)
			{
				printf("System gnuplot failed!\n");
			}
		}
		else
		{
			if (use_actual_piece_artwork == 1)
			{
				systemreturn = system("gnuplot plotscripts/chessboard_simulation_artwork.gplot");
			}
			else
			{
				systemreturn = system("gnuplot plotscripts/chessboard_simulation_basic.gplot");
			}

			if (systemreturn == -1)
			{
				printf("System gnuplot failed!\n");
			}
		}
	}
	else
	{
		if (use_actual_piece_artwork == 1)
		{
			systemreturn = system("gnuplot plotscripts/chessboard_artwork.gplot");
		}
		else
		{
			systemreturn = system("gnuplot plotscripts/chessboard_basic.gplot");
		}

		if (systemreturn == -1)
		{
			printf("System gnuplot failed!\n");
		}

		if (use_actual_piece_artwork == 1)
		{
			systemreturn = system("gnuplot plotscripts/chessboard_simulation_artwork.gplot");
		}
		else
		{
			systemreturn = system("gnuplot plotscripts/chessboard_simulation_basic.gplot");
		}

		if (systemreturn == -1)
		{
			printf("System gnuplot failed!\n");
		}
	}
} // end of PrintPieces function

/******************************************************************************************************/
/******************************************** PIECE VALUES ********************************************/
/******************************************************************************************************/

/* Initialize piece and position values to be used during board evaluation */
void InitializePieceAndPositionValues(int **piece_value, int ***non_passed_pawn_opening_values, int ***non_passed_pawn_end_game_values, double ***pawn_advancement_multipliers, double ***knight_position_modifiers)
{
	unsigned int i, j;
	int systemreturn;

	(*piece_value) = malloc(sizeof(int) * chess_constants.number_of_unique_pieces);
	(*piece_value)[0] = 100;
	(*piece_value)[1] = 320;
	(*piece_value)[2] = 333;
	(*piece_value)[3] = 510;
	(*piece_value)[4] = 880;
	(*piece_value)[5] = 1000000;

	(*non_passed_pawn_opening_values) = malloc(sizeof(int*) * chess_constants.number_of_columns);
	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		(*non_passed_pawn_opening_values)[i] = malloc(sizeof(int) * chess_constants.number_of_rows);
	} // end of i loop

	(*non_passed_pawn_end_game_values) = malloc(sizeof(int*) * chess_constants.number_of_columns);
	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		(*non_passed_pawn_end_game_values)[i] = malloc(sizeof(int) * chess_constants.number_of_rows);
	} // end of i loop

	FILE *infile_non_passed_pawn_values = fopen("inputs/non_passed_pawn_values.txt", "r"); // read only
	for (i = 0; i < chess_constants.number_of_rows; i++)
	{
		for (j = 0; j < chess_constants.number_of_columns; j++)
		{
			systemreturn = fscanf(infile_non_passed_pawn_values, "%d\t", &(*non_passed_pawn_opening_values)[j][i]); // opening
			if (systemreturn == -1)
			{
				printf("Reading inputs/non_passed_pawn_values.txt failed!\n");
			}
		} // end of j loop
	} // end of i loop

	for (i = 0; i < chess_constants.number_of_rows; i++)
	{
		for (j = 0; j < chess_constants.number_of_columns; j++)
		{
			systemreturn = fscanf(infile_non_passed_pawn_values, "%d\t", &(*non_passed_pawn_end_game_values)[j][i]); // end game
			if (systemreturn == -1)
			{
				printf("Reading inputs/non_passed_pawn_values.txt failed!\n");
			}
		} // end of j loop
	} // end of i loop
	fclose(infile_non_passed_pawn_values);

	(*pawn_advancement_multipliers) = malloc(sizeof(double*) * 4);
	for (i = 0; i < 4; i++)
	{
		(*pawn_advancement_multipliers)[i] = malloc(sizeof(double) * 4);
	} // end of i loop

	FILE *infile_pawn_advancement_multipliers = fopen("inputs/pawn_advancement_multipliers.txt", "r"); // read only
	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			systemreturn = fscanf(infile_pawn_advancement_multipliers, "%lf\t", &(*pawn_advancement_multipliers)[j][i]);
			if (systemreturn == -1)
			{
				printf("Reading inputs/pawn_advancement_multipliers.txt failed!\n");
			}
		} // end of j loop
	} // end of i loop
	fclose(infile_pawn_advancement_multipliers);

	(*knight_position_modifiers) = malloc(sizeof(double*) * chess_constants.number_of_columns);
	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		(*knight_position_modifiers)[i] = malloc(sizeof(double) * chess_constants.number_of_rows);
	} // end of i loop

	FILE *infile_knight_position_modifiers = fopen("inputs/knight_position_modifiers.txt", "r"); // read only
	for (i = 0; i < chess_constants.number_of_rows; i++)
	{
		for (j = 0; j < chess_constants.number_of_columns; j++)
		{
			systemreturn = fscanf(infile_knight_position_modifiers, "%lf\t", &(*knight_position_modifiers)[j][i]);
			if (systemreturn == -1)
			{
				printf("Reading inputs/knight_position_modifiers.txt failed!\n");
			}
		} // end of j loop
	} // end of i loop
	fclose(infile_knight_position_modifiers);
} // end of InitializePieceAndPositionValues function

/******************************************************************************************************/
/********************************************* GAME SETUP *********************************************/
/******************************************************************************************************/

/* This function sets up some of the initial game tracking arrays */
void GameSetup(int board_history_length, int **square, int **piece_coordinates, int **move_coordinates, int ****board_history)
{
	unsigned int i, j, k;
	int systemreturn;

	(*piece_coordinates) = malloc(sizeof(int) * 2);
	(*move_coordinates) = malloc(sizeof(int) * 2);

	(*board_history) = malloc(sizeof(int**) * chess_constants.number_of_columns);
	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		(*board_history)[i] = malloc(sizeof(int*) * chess_constants.number_of_rows);
		for (j = 0; j < chess_constants.number_of_rows; j++)
		{
			(*board_history)[i][j] = malloc(sizeof(int) * board_history_length);
			for (k = 0; k < board_history_length; k++)
			{
				(*board_history)[i][j][k] = 0;
			} // end of k loop

			(*board_history)[i][j][0] = square[i][j];
		} // end of j loop
	} // end of i loop

	systemreturn = system("./delete_chess_boards.sh");
	if (systemreturn == -1)
	{
		printf("System delete_chess_boards.sh failed!\n");
	}
} // end of GameSetup function

/******************************************************************************************************/
/******************************************* BEGIN PLAYING ********************************************/
/******************************************************************************************************/

/* This function gets the number of players from user */
void GetNumberOfPlayers(struct PlayerOptions *player_options)
{
	int systemreturn;
	char player_char, extra_char;

	do
	{
		printf("Do you want to simulate, play against computer, or do two player (0, 1, 2)?\n");

		systemreturn = scanf(" %c", &player_char);
		if (systemreturn == -1)
		{
			printf("GetNumberOfPlayers: Failed reading player_char\n");
		}
		while ((extra_char = fgetc(stdin)) != '\n' && extra_char != EOF); /* Flush stdin */

		if (player_char == '0')
		{
			player_options->players = 0;
		}
		else if (player_char == '1')
		{
			player_options->players = 1;
		}
		else if (player_char == '2')
		{
			player_options->players = 2;
		}
	} while ((player_options->players < 0) || (player_options->players > 2));
} // end of GetNumberOfPlayers function

/******************************************************************************************************/
/****************************************** ADD COMPUTER(S) *******************************************/
/******************************************************************************************************/

/* This function gets the computer's difficulty from the user */
void GetComputerDifficulty(unsigned int maximum_depth, int computer, int *difficulty_computer)
{
	int atoi_return, systemreturn;
	char difficulty_char, extra_char;

	do
	{
		printf("What difficulty do you want computer %d to be (1 to %d)?\n", computer, maximum_depth);

		systemreturn = scanf(" %c", &difficulty_char);
		if (systemreturn == -1)
		{
			printf("GetComputerDifficulty: Failed reading difficulty_char\n");
		}
		while ((extra_char = fgetc(stdin)) != '\n' && extra_char != EOF); /* Flush stdin */

		atoi_return = atoi(&difficulty_char);

		if (atoi_return > 0)
		{
			(*difficulty_computer) = atoi_return;
		}

	} while (((*difficulty_computer) < 1) || ((*difficulty_computer) > maximum_depth));

	printf("Computer %d difficulty set to %d\n", computer, (*difficulty_computer));
} // end of GetComputerDifficulty function

/* This function gets the color choice from the user for single player games */
void GetSinglePlayerColorChoice(struct PlayerOptions *player_options)
{
	int systemreturn;
	char color_char, extra_char;

	do
	{
		printf("Do you want to be white or black (w or b)?\n");

		systemreturn = scanf(" %c", &color_char);
		if (systemreturn == -1)
		{
			printf("GetSinglePlayerColorChoice: Failed reading color_char\n");
		}
		while ((extra_char = fgetc(stdin)) != '\n' && extra_char != EOF); /* Flush stdin */

		if (color_char == 'w' || color_char == 'W')
		{
			player_options->color_choice = 0;
		}
		else if (color_char == 'b' || color_char == 'B')
		{
			player_options->color_choice = 1;
		}
	} while ((player_options->color_choice < 0) || (player_options->color_choice > 1));

	if (player_options->color_choice == 0) // if player 1 is white
	{
		printf("You are playing as white!\n");
	} // end of if player 1 is white
	else // if player 1 is black
	{
		printf("You are playing as black!\n");
	} // end of if player 1 is black
} // end of GetSinglePlayerColorChoice function

/* This function creates computer only variables for logging */
void CreateComputerOnlyLoggingVariables(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, unsigned int maximum_depth, int *depth_daughters, int *depth_valid_move_count, int ***piece_board_history, int ***move_board_history)
{
	unsigned int i, j;

	if (activate_computer_move_log == 1)
	{
		fprintf(outfile_computer_move_log, "turn\tdepth\tpiece_file\tpiece_rank\tpiecesquare\tmove_file\tmove_rank\tmovesquare\t");
		fprintf(outfile_computer_move_log, "white_queen_side_castle\twhite_king_side_castle\tblack_queen_side_castle\tblack_king_side_castle\twhite_queen_side_castle_allow\twhite_king_side_castle_allow\tblack_queen_side_castle_allow\tblack_king_side_castle_allow\ten_passant_captured\twhitepawn_en_passant_status\tblackpawn_en_passant_status\tpromotion");
		for (i = 0; i < maximum_depth; i++)
		{
			fprintf(outfile_computer_move_log, "\tdepth_valid_move_count[%u]", i);
		} // end of i loop
		fprintf(outfile_computer_move_log, "\n");
	}

	if (activate_computer_move_score_log == 1)
	{
		fprintf(outfile_computer_move_score_log, "turn\tdepth\tpiece_file\tpiece_rank\tpiecesquare\tmove_file\tmove_rank\tmovesquare\tscore\t");
		fprintf(outfile_computer_move_score_log, "white_queen_side_castle\twhite_king_side_castle\tblack_queen_side_castle\tblack_king_side_castle\twhite_queen_side_castle_allow\twhite_king_side_castle_allow\tblack_queen_side_castle_allow\tblack_king_side_castle_allow\ten_passant_captured\twhitepawn_en_passant_status\tblackpawn_en_passant_status\tpromotion");
		for (i = 0; i < maximum_depth; i++)
		{
			fprintf(outfile_computer_move_score_log, "\tdepth_valid_move_count[%u]", i);
		} // end of i loop
		fprintf(outfile_computer_move_score_log, "\n");
	}

	/* Initialize to zero */
	for (i = 0; i < maximum_depth; i++)
	{
		depth_daughters[i] = 0;
		depth_valid_move_count[i] = 0;
	} // end of i loop

	(*piece_board_history) = malloc(sizeof(int*) * maximum_depth);
	for (i = 0; i < maximum_depth; i++)
	{
		(*piece_board_history)[i] = malloc(sizeof(int) * 3);
		for (j = 0; j < 3; j++)
		{
			(*piece_board_history)[i][j] = -9;
		} // end of j loop
	} // end of i loop

	(*move_board_history) = malloc(sizeof(int*) * maximum_depth);
	for (i = 0; i < maximum_depth; i++)
	{
		(*move_board_history)[i] = malloc(sizeof(int) * 3);
		for (j = 0; j < 3; j++)
		{
			(*move_board_history)[i][j] = -9;
		} // end of j loop
	} // end of i loop

	if (activate_board_history_log == 1)
	{
		fprintf(outfile_piece_move_board_history, "turn\tdepth\tscore");
		for (i = 0; i < maximum_depth; i++)
		{
			fprintf(outfile_piece_move_board_history, "\tpiece_file[%u]\tpiece_rank[%u]\tpiecesquare[%u]\tmove_file[%u]\tmove_rank[%u]\tmovesquare[%u]", i, i, i, i, i, i);
		} // end of i loop
		fprintf(outfile_piece_move_board_history, "\n");
	}
} // end of CreateComputerOnlyLoggingVariables function

/******************************************************************************************************/
/********************************************* NEW TURN ***********************************************/
/******************************************************************************************************/

/* This function starts a new turn */
void StartNewTurn(FILE *outfile_turn, struct BoardState *board_state, unsigned int **piece_count, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status)
{
	board_state->turn++;

	outfile_turn = fopen("turn.txt", "w");
	fprintf(outfile_turn, "%d\n", board_state->turn);
	fclose(outfile_turn);

	printf("\nTurn %d: ", board_state->turn);
	printf("White = %u, WPawn = %u, WKnight = %u, WBishop = %u, WRook = %u, WQueen = %u, WKing = %u, ", board_state->white_pieces, piece_count[0][0], piece_count[0][1], piece_count[0][2], piece_count[0][3], piece_count[0][4], piece_count[0][5]);
	printf("Black = %u, BPawn = %u, BKnight = %u, BBishop = %u, BRook = %u, BQueen = %u, BKing = %u\n", board_state->black_pieces, piece_count[1][0], piece_count[1][1], piece_count[1][2], piece_count[1][3], piece_count[1][4], piece_count[1][5]);

	ResetCoordinatesAndPawnEnPassantNewTurn(board_state, piece_coordinates, move_coordinates, pawn_en_passant_status);

	PrintBoardState(board_state);

	if (board_state->white_in_check == 1) // if white is in check
	{
		printf("Start of turn: White is in check!\n");
	} // end of if white is in check

	if (board_state->black_in_check == 1) // if black is in check
	{
		printf("Start of turn: Black is in check!\n");
	} // end of if black is in check
} // end of StartNewTurn function

/* This function resets the chosen coordinates and pawn en passant status for the new turn */
void ResetCoordinatesAndPawnEnPassantNewTurn(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status)
{
	unsigned int i;

	for (i = 0; i < 2; i++)
	{
		piece_coordinates[i] = -9;
		move_coordinates[i] = -9;
	} // end of i loop

	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		pawn_en_passant_status[1 - board_state->turn % 2][i] = 0;
	} // end of i loop
} // end of ResetCoordinatesAndPawnEnPassantNewTurn function

/* This function prints the current values of a BoardState */
void PrintBoardState(struct BoardState *board_state)
{
	printf("PrintBoardState: turn = %u\n", board_state->turn);
	printf("PrintBoardState: white_pieces = %u, black_pieces = %u, white_pieces_old = %u, black_pieces_old = %u\n", board_state->white_pieces, board_state->black_pieces, board_state->white_pieces_old, board_state->black_pieces_old);
	printf("PrintBoardState: white_in_check = %d, black_in_check = %d\n", board_state->white_in_check, board_state->black_in_check);
	printf("PrintBoardState: white_queen_side_castle = %d, black_queen_side_castle = %d, white_king_side_castle = %d, black_king_side_castle = %d\n", board_state->white_queen_side_castle, board_state->black_queen_side_castle, board_state->white_king_side_castle, board_state->black_king_side_castle);
	printf("PrintBoardState: white_queen_side_castle_allow = %d, black_queen_side_castle_allow = %d, white_king_side_castle_allow = %d, black_king_side_castle_allow = %d\n", board_state->white_queen_side_castle_allow, board_state->black_queen_side_castle_allow, board_state->white_king_side_castle_allow, board_state->black_king_side_castle_allow);
	printf("PrintBoardState: en_passant_captured = %d\n", board_state->en_passant_captured);
	printf("PrintBoardState: pawn_move = %u, capture_move = %u\n", board_state->pawn_move, board_state->capture_move);
	printf("PrintBoardState: closed_position = %d, end_game = %d\n", board_state->closed_position, board_state->end_game);
	printf("PrintBoardState: evaluation_score = %d, game_over = %d\n\n", board_state->evaluation_score, board_state->game_over);
} // end of PrintBoardState function

/* This function prints the evaluation score based on position and game phase */
void PrintPositionGamePhaseScore(struct BoardState *board_state)
{
	if (board_state->closed_position == 0) // if open position
	{
		if (board_state->end_game == 0) // if in opening
		{
			printf("Open Position Board Opening Evaluation Score = %d\n", board_state->evaluation_score);
		} // end of if in opening
		else // if in end_game
		{
			printf("Open Position Board Endgame Evaluation Score = %d\n", board_state->evaluation_score);
		} // end of if in end_game
	} // end of if open position
	else // if closed position
	{
		if (board_state->end_game == 0) // if in opening
		{
			printf("Closed Position Board Opening Evaluation Score = %d\n", board_state->evaluation_score);
		} // end of if in opening
		else // if in end_game
		{
			printf("Closed Position Board Endgame Evaluation Score = %d\n", board_state->evaluation_score);
		} // end of if in end_game
	} // end of if closed position
} // end of PrintPositionGamePhaseScore function

/******************************************************************************************************/
/******************************************** HUMAN TURN **********************************************/
/******************************************************************************************************/

/* This function executes a human turn */
int HumanTurn(struct PlayerOptions *player_options, struct BoardState *board_state, char *basic_chessboard, int **square, int **pawn_en_passant_status, unsigned int **piece_count, char *read_coordinates, int *piece_coordinates, int *move_coordinates, int ****piece_positions, int **color, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history)
{
	int skip = 0, invalid_move = 1;

	//---------------------------------------------------------------------Piece & Move Selection---------------------------------------------------------------------
	skip = SelectPieceCoordinates(board_state, read_coordinates, piece_coordinates, square, piece_positions);

	if (skip == 0) // if we aren't skipping
	{
		PrintPieceSelected(square, piece_coordinates);

		skip = SelectMoveCoordinates(board_state, read_coordinates, move_coordinates, piece_coordinates, square);
	} // end of if we aren't skipping

	//---------------------------------------------------------------------Movement Validation---------------------------------------------------------------------
	if (skip == 0) // if we aren't skipping
	{
		invalid_move = CheckHumanMoveValidity(board_state, square, read_coordinates, piece_coordinates, move_coordinates, pawn_en_passant_status, piece_positions, &skip);

		//---------------------------------------------------------------------Perform Moves---------------------------------------------------------------------
		if (invalid_move == 0 && skip == 0) // if move is valid and we aren't skipping
		{
			if (PerformHumanMoves(player_options, board_state, basic_chessboard, board_state->turn, square, pawn_en_passant_status, piece_count, piece_coordinates, move_coordinates, piece_positions, color, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history) == 1)
			{
				skip = 2; // exit since game is over
			}
		} // end of if move is valid and we aren't skipping
	} // end of if we aren't skipping

	return skip;
} // end of HumanTurn function

/* This function selects piece coordinates */
int SelectPieceCoordinates(struct BoardState *board_state, char *read_coordinates, int *piece_coordinates, int **square, int ****piece_positions)
{
	int skip = -1, systemreturn;
	char extra_char;

	if (board_state->turn % 2 == 1) // if it's white's turn
	{
		printf("It is white's turn! Select a white piece (column, row) or x to exit!\n");
	} // end of if it's white's turn
	else // if it's black's turn
	{
		printf("It is black's turn! Select a black piece (column, row) or x to exit!\n");
	} // end of if it's black's turn

	do
	{
		systemreturn = scanf(" %c", &read_coordinates[0]);
		if (systemreturn == -1)
		{
			printf("SelectPieceCoordinates: Failed reading read_coordinates[0]\n");
		}

		if (read_coordinates[0] == 'x') // if want to exit program
		{
			printf("Exiting program!\n");
			skip = 2;
		} // end of if want to exit program
		else // if DON'T want to exit program
		{
			systemreturn = scanf("%c", &read_coordinates[1]);
			if (systemreturn == -1)
			{
				printf("SelectPieceCoordinates: Failed reading read_coordinates[1]\n");
			}
			while ((extra_char = fgetc(stdin)) != '\n' && extra_char != EOF); /* Flush stdin */

			ConvertReadColumnCoordinatesChartoInt(read_coordinates, piece_coordinates);
			ConvertReadRowCoordinatesChartoInt(read_coordinates, piece_coordinates);

			if ((piece_coordinates[0] < 0) || (piece_coordinates[0] > 7) || (piece_coordinates[1] < 0) || (piece_coordinates[1] > 7)) // if piece coordinate is not even on the board
			{
				printf("(%d, %d) is not on the board! Choose again (column, row) or x to exit!\n", piece_coordinates[0], piece_coordinates[1]);
			} // end of if piece coordinate is not even on the board
			else if (square[piece_coordinates[0]][piece_coordinates[1]] < 0) // if piece coordinate is a black piece
			{
				if (board_state->turn % 2 == 1) // if it's white's turn
				{
					printf("(%d, %d) is a black piece NOT a white piece! Try again (column, row) or x to exit!\n", piece_coordinates[0], piece_coordinates[1]);
				} // end of if it's white's turn
				else // if it's black's turn
				{
					skip = 0;
				} // end of if it's black's turn
			} // end of if piece coordinate is a black piece
			else if (square[piece_coordinates[0]][piece_coordinates[1]] > 0) // if piece coordinate is a white piece
			{
				if (board_state->turn % 2 == 0) // if it's black's turn
				{
					printf("(%d, %d) is a white piece, NOT a black piece! Try again (column, row) or x to exit!\n", piece_coordinates[0], piece_coordinates[1]);
				} // end of if it's black's turn
				else // if it is white's turn
				{
					skip = 0;
				} // end of if it is white's turn
			} // end of if piece coordinate is a white piece
			else if (square[piece_coordinates[0]][piece_coordinates[1]] == 0)
			{
				printf("(%d, %d) is an empty square! Try again (column, row) or x to exit!\n", piece_coordinates[0], piece_coordinates[1]);
			}
		} // end of if DON'T want to exit program
	} while (skip == -1);

	return skip;
} // end of SelectPieceCoordinates function

/* This function converts column coordinates from char to int */
void ConvertReadColumnCoordinatesChartoInt(char *read_coordinates, int *variable_coordinates)
{
	if (read_coordinates[0] == 'a' || read_coordinates[0] == 'A')
	{
		variable_coordinates[0] = 0;
	}
	else if (read_coordinates[0] == 'b' || read_coordinates[0] == 'B')
	{
		variable_coordinates[0] = 1;
	}
	else if (read_coordinates[0] == 'c' || read_coordinates[0] == 'C')
	{
		variable_coordinates[0] = 2;
	}
	else if (read_coordinates[0] == 'd' || read_coordinates[0] == 'D')
	{
		variable_coordinates[0] = 3;
	}
	else if (read_coordinates[0] == 'e' || read_coordinates[0] == 'E')
	{
		variable_coordinates[0] = 4;
	}
	else if (read_coordinates[0] == 'f' || read_coordinates[0] == 'F')
	{
		variable_coordinates[0] = 5;
	}
	else if (read_coordinates[0] == 'g' || read_coordinates[0] == 'G')
	{
		variable_coordinates[0] = 6;
	}
	else if (read_coordinates[0] == 'h' || read_coordinates[0] == 'H')
	{
		variable_coordinates[0] = 7;
	}
	else
	{
		variable_coordinates[0] = -1;
	}
} // end of ConvertReadColumnCoordinatesChartoInt function

/* This function converts row coordinates from char to int */
void ConvertReadRowCoordinatesChartoInt(char *read_coordinates, int *variable_coordinates)
{
	if (read_coordinates[1] == '1')
	{
		variable_coordinates[1] = 0;
	}
	else if (read_coordinates[1] == '2')
	{
		variable_coordinates[1] = 1;
	}
	else if (read_coordinates[1] == '3')
	{
		variable_coordinates[1] = 2;
	}
	else if (read_coordinates[1] == '4')
	{
		variable_coordinates[1] = 3;
	}
	else if (read_coordinates[1] == '5')
	{
		variable_coordinates[1] = 4;
	}
	else if (read_coordinates[1] == '6')
	{
		variable_coordinates[1] = 5;
	}
	else if (read_coordinates[1] == '7')
	{
		variable_coordinates[1] = 6;
	}
	else if (read_coordinates[1] == '8')
	{
		variable_coordinates[1] = 7;
	}
	else
	{
		variable_coordinates[1] = -1;
	}
} // end of ConvertReadRowCoordinatesChartoInt function

/* This function prints piece selected */
void PrintPieceSelected(int **square, int *piece_coordinates)
{
	/* Color */
	if (square[piece_coordinates[0]][piece_coordinates[1]] > 0)
	{
		printf("White ");
	}
	else
	{
		printf("Black ");
	}

	/* Piece type */
	if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 1)
	{
		printf("pawn selected!\n");
	}
	else if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 2)
	{
		printf("knight selected!\n");
	}
	else if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 3)
	{
		printf("bishop selected!\n");
	}
	else if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 4)
	{
		printf("rook selected!\n");
	}
	else if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 5)
	{
		printf("queen selected!\n");
	}
	else if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 6)
	{
		printf("king selected!\n");
	}
} // end of PrintPieceSelected function

/* This function selects move coordinates */
int SelectMoveCoordinates(struct BoardState *board_state, char *read_coordinates, int *move_coordinates, int *piece_coordinates, int **square)
{
	int skip = -1, systemreturn;
	char extra_char;

	printf("Where do you want to move selected piece (column, row), p to choose a different piece, or x to exit?\n");

	do
	{
		systemreturn = scanf(" %c", &read_coordinates[0]);
		if (systemreturn == -1)
		{
			printf("SelectMoveCoordinates: Failed reading read_coordinates[1]\n");
		}

		if (read_coordinates[0] == 'x') // if want to exit program
		{
			printf("Exiting program!\n");
			skip = 2;
		} // end of if want to exit program
		else if (read_coordinates[0] == 'p') // if want to choose a different piece
		{
			board_state->turn--;
			skip = 1;
		} // end of if want to choose a different piece
		else // if DON'T want to exit program or choose a different piece
		{
			systemreturn = scanf("%c", &read_coordinates[1]);
			if (systemreturn == -1)
			{
				printf("SelectMoveCoordinates: Failed reading read_coordinates[1]\n");
			}
			while ((extra_char = fgetc(stdin)) != '\n' && extra_char != EOF); /* Flush stdin */

			ConvertReadColumnCoordinatesChartoInt(read_coordinates, move_coordinates);
			ConvertReadRowCoordinatesChartoInt(read_coordinates, move_coordinates);

			if ((move_coordinates[0] < 0) || (move_coordinates[0] > 7) || (move_coordinates[1] < 0) || (move_coordinates[1] > 7)) // if move coordinate is not even on the board
			{
				printf("(%d, %d) is not on the board! Choose again (column, row) or x to exit!\n", move_coordinates[0], move_coordinates[1]);
			} // end of if move coordinate is not even on the board
			else if (square[move_coordinates[0]][move_coordinates[1]] < 0) // if move coordinate is a black piece
			{
				if (board_state->turn % 2 == 1) // if it's white's turn
				{
					skip = 0;
				} // end of if it's white's turn
				else // if it's black's turn
				{
					printf("(%d, %d) is your own black piece! Black can't move to black occupied square. Try again (column, row) or x to exit!\n", move_coordinates[0], move_coordinates[1]);
				} // end of if it's black's turn
			} // end of if move coordinate is a black piece
			else if (square[move_coordinates[0]][move_coordinates[1]] > 0) // if move coordinate is a white piece
			{
				if (board_state->turn % 2 == 0) // if it's black's turn
				{
					skip = 0;
				} // end of if it's black's turn
				else // if it is white's turn
				{
					printf("(%d, %d) is your own white piece! White can't move to white occupied square. Try again (column, row) or x to exit!\n", move_coordinates[0], move_coordinates[1]);
				} // end of if it is white's turn
			} // end of if move coordinate is a white piece
			else if (square[move_coordinates[0]][move_coordinates[1]] == 0)
			{
				skip = 0;
			}
		} // end of if DON'T want to exit program or choose a different piece
	} while (skip == -1);

	return skip;
} // end of SelectMoveCoordinates function

/* This function checks the validity of human move */
int CheckHumanMoveValidity(struct BoardState *board_state, int **square, char *read_coordinates, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int ****piece_positions, int *skip)
{
	int invalid_move = 1;

	while (invalid_move == 1 && (*skip) == 0)
	{
		if ((*skip) == 0) // if we aren't skipping
		{
			if (board_state->turn % 2 == 1) // if it is white's turn
			{
				invalid_move = MovementValidation(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, piece_positions, 1);

				if (invalid_move == 1) // if move is still invalid
				{
					if (board_state->white_in_check == 0) // if white is NOT in check
					{
						WhiteQueenSideCastling(board_state, piece_coordinates, move_coordinates, square, &invalid_move);

						if (invalid_move == 1) // if move is still invalid
						{
							WhiteKingSideCastling(board_state, piece_coordinates, move_coordinates, square, &invalid_move);
						} // end of if move is still invalid
					} // end of if white is NOT in check
				} // end of if move is still invalid
			} // end of if it is white's turn
			else // if it is black's turn
			{
				if (invalid_move == 1) // if move is still invalid
				{
					invalid_move = MovementValidation(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, piece_positions, 1);

					if (board_state->black_in_check == 0) // if black is NOT in check
					{
						BlackQueenSideCastling(board_state, piece_coordinates, move_coordinates, square, &invalid_move);

						if (invalid_move == 1) // if move is still invalid
						{
							BlackKingSideCastling(board_state, piece_coordinates, move_coordinates, square, &invalid_move);
						} // end of if move is still invalid
					} // end of if black is NOT in check
				} // end of if move is still invalid
			} // end of if it is black's turn
		} // end of if we aren't skipping

		if (invalid_move == 1) // if move is still invalid
		{
			(*skip) = SelectMoveCoordinates(board_state, read_coordinates, move_coordinates, piece_coordinates, square);
		} // end of if move is still invalid
	} // end of while

	return invalid_move;
} // end of CheckHumanMoveValidity function

/******************************************************************************************************/
/*************************************** MOVEMENT VALIDATION ******************************************/
/******************************************************************************************************/

/* This function validates movements */
int MovementValidation(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int ****piece_positions, int human)
{
	int invalid_move = 1, check = 0, not_same_color_or_king = 0;

	if (square[piece_coordinates[0]][piece_coordinates[1]] > 0) // if piece is white
	{
		if ((square[move_coordinates[0]][move_coordinates[1]] <= 0) && (square[move_coordinates[0]][move_coordinates[1]] != -6)) // if move square isn't white or black king
		{
			not_same_color_or_king = 1;

			if (square[piece_coordinates[0]][piece_coordinates[1]] == 1) // if piece is white pawn
			{
				WhitePawnMovement(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, &invalid_move, 0);
			} // end of if piece is white pawn
		} // end of if move square isn't white or black king
	}
	else if (square[piece_coordinates[0]][piece_coordinates[1]] < 0) // if piece is black
	{
		if ((square[move_coordinates[0]][move_coordinates[1]] >= 0) && (square[move_coordinates[0]][move_coordinates[1]] != 6)) // if move square isn't black or white king
		{
			not_same_color_or_king = 1;

			if (square[piece_coordinates[0]][piece_coordinates[1]] == -1) // if piece is black pawn
			{
				BlackPawnMovement(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, &invalid_move, 0);
			} // end of if piece is black pawn
		} // end of if move square isn't black or white king
	}

	if (not_same_color_or_king == 1)
	{
		if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 2) // if piece is knight
		{
			KnightMovement(square, piece_coordinates, move_coordinates, &invalid_move);
		} // end of if piece is knight
		else if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 3) // if piece is bishop
		{
			BishopMovement(square, piece_coordinates, move_coordinates, &invalid_move);
		} // end of if piece is bishop
		else if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 4) // if piece is rook
		{
			RookMovement(square, piece_coordinates, move_coordinates, &invalid_move);
		} // end of if piece is rook
		else if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 5) // if piece is queen
		{
			QueenMovement(square, piece_coordinates, move_coordinates, &invalid_move);
		} // end of if piece is queen
		else if (abs(square[piece_coordinates[0]][piece_coordinates[1]]) == 6)// if piece is king
		{
			KingMovement(square, piece_coordinates, move_coordinates, &invalid_move);
		} // end of if piece is king
	}

	if (invalid_move == 1)
	{
		if (human == 1)
		{
			printf("Move is invalid! Please choose another!\n");
		}
	}
	else if (invalid_move == 0)
	{
		check = CheckValidMovesForCheck(piece_coordinates, move_coordinates, square, piece_positions);

		if (check == 1)
		{
			if (human == 1)
			{
				printf("Move is valid but it would put you in check! Please choose another!\n");
			}
			invalid_move = 1;
		}
		else
		{
			if (human == 1)
			{
				printf("Move is valid!\n");
			}
		}
	}

	return invalid_move;
} // end of MovementValidation function

/* This function checks if white pawn can move */
void WhitePawnMovement(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int *invalid_move, int mate_checking)
{
	if (square[move_coordinates[0]][move_coordinates[1]] == 0) // if square isn't occupied
	{
		if (move_coordinates[0] == piece_coordinates[0]) // if same column
		{
			if (piece_coordinates[1] == 1) // if in starting row
			{
				if ((move_coordinates[1] == piece_coordinates[1] + 2) && (square[piece_coordinates[0]][piece_coordinates[1] + 1] == 0)) // if move up two rows
				{
					(*invalid_move) = 0;
				} // end of if move up two rows
				else if (move_coordinates[1] == piece_coordinates[1] + 1) // if move up a row
				{
					(*invalid_move) = 0;
				} // end of if move up a row
			} // end of if in starting row
			else // if NOT in starting row
			{
				if (move_coordinates[1] == piece_coordinates[1] + 1) // if move up a row
				{
					(*invalid_move) = 0;
				} // end of if move up a row
			} // end of if NOT in starting row
		} // end of if same column
		else // if different column
		{
			if (move_coordinates[1] == 5) // if white pawn moving to row right below black pawn row
			{
				if (piece_coordinates[1] == 4) // if white pawn is 2 rows below black pawn starting row
				{
					if (square[move_coordinates[0]][move_coordinates[1] - 1] == -1) // if square in row below where white pawn is moving is occupied by a black pawn
					{
						if (piece_coordinates[0] - 1 >= 0 && pawn_en_passant_status[1][piece_coordinates[0] - 1] == 1) // if square has black pawn that is en passant
						{
							if (mate_checking == 0)
							{
								board_state->en_passant_captured = piece_coordinates[0] - 1;
							}
							(*invalid_move) = 0;
						} // end of if square has black pawn that is en passant
						else if (piece_coordinates[0] + 1 <= 7 && pawn_en_passant_status[1][piece_coordinates[0] + 1] == 1) // if square has black pawn that is en passant
						{
							if (mate_checking == 0)
							{
								board_state->en_passant_captured = piece_coordinates[0] + 1;
							}
							(*invalid_move) = 0;
						} // end of if square has black pawn that is en passant
					} // end of if square in row below where white pawn is moving is occupied by a black pawn
				} // end of if white pawn is 2 rows below black pawn starting row
			} // end of if white pawn moving to row right below black pawn row
		} // end of if different column
	} // end of if square isn't occupied
	else if ((square[move_coordinates[0]][move_coordinates[1]] < 0) && (square[move_coordinates[0]][move_coordinates[1]] != -6)) // if square has black piece besides king
	{
		if ((move_coordinates[0] == piece_coordinates[0] - 1) || (move_coordinates[0] == piece_coordinates[0] + 1)) // if move one column over
		{
			if (move_coordinates[1] == piece_coordinates[1] + 1) // if move up a row
			{
				(*invalid_move) = 0;
			} // end of if move up a row
		} // end of if move one column over
	} // end of if square has black piece besides king
} // end of whitePawn function

/* This function checks if black pawn can move */
void BlackPawnMovement(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int *invalid_move, int mate_checking)
{
	if (square[move_coordinates[0]][move_coordinates[1]] == 0) // if square isn't occupied
	{
		if (move_coordinates[0] == piece_coordinates[0]) // if same column
		{
			if (piece_coordinates[1] == 6) // if in starting row
			{
				if ((move_coordinates[1] == piece_coordinates[1] - 2) && (square[piece_coordinates[0]][piece_coordinates[1] - 1] == 0)) // if move down two rows
				{
					(*invalid_move) = 0;
				} // end of if move down two rows
				else if (move_coordinates[1] == piece_coordinates[1] - 1) // if move down a row
				{
					(*invalid_move) = 0;
				} // end of if move down a row
			} // end of if in starting row
			else // if NOT in starting row
			{
				if (move_coordinates[1] == piece_coordinates[1] - 1) // if move down a row
				{
					(*invalid_move) = 0;
				} // end of if move down a row
			} // end of if NOT in starting row
		} // end of if same column
		else // if different column
		{
			if (move_coordinates[1] == 2) // if black pawn moving to row right above white pawn row
			{
				if (piece_coordinates[1] == 3) // if black pawn is 2 rows above white pawn starting row
				{
					if (square[move_coordinates[0]][move_coordinates[1] + 1] == 1) // if square in row above where black pawn is moving is occupied by a white pawn
					{
						if (piece_coordinates[0] - 1 >= 0 && pawn_en_passant_status[0][piece_coordinates[0] - 1] == 1) // if square has white pawn that is en passant
						{
							if (mate_checking == 0)
							{
								board_state->en_passant_captured = piece_coordinates[0] - 1;
							}
							(*invalid_move) = 0;
						} // end of if square has white pawn that is en passant
						else if (piece_coordinates[0] + 1 <= 7 && pawn_en_passant_status[0][piece_coordinates[0] + 1] == 1) // if square has white pawn that is en passant
						{
							if (mate_checking == 0)
							{
								board_state->en_passant_captured = piece_coordinates[0] + 1;
							}
							(*invalid_move) = 0;
						} // end of if square has white pawn that is en passant
					} // end of if square in row above where black pawn is moving is occupied by a white pawn
				} // if black pawn is 2 rows above white pawn starting row
			} // end of if black pawn moving to row right above white pawn row
		} // end of if different column
	} // end of if square isn't occupied
	else if ((square[move_coordinates[0]][move_coordinates[1]] > 0) && (square[move_coordinates[0]][move_coordinates[1]] != 6)) // if square has white piece besides king
	{
		if ((move_coordinates[0] == piece_coordinates[0] - 1) || (move_coordinates[0] == piece_coordinates[0] + 1)) // if move one column over
		{
			if (move_coordinates[1] == piece_coordinates[1] - 1) // if move down a row
			{
				(*invalid_move) = 0;
			} // end of if move down a row
		} // end of if move one column over
	} // end of if square has white piece besides king
} // end of blackPawn function

/* This function checks if knight can move */
void KnightMovement(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move)
{
	if ((abs(move_coordinates[1] - piece_coordinates[1]) == 1) && (abs(move_coordinates[0] - piece_coordinates[0]) == 2)) // if moving up or down a row and moving right or left 2 columns
	{
		(*invalid_move) = 0;
	} // end of if moving up or down a row and moving right or left 2 columns
	else if ((abs(move_coordinates[1] - piece_coordinates[1]) == 2) && (abs(move_coordinates[0] - piece_coordinates[0]) == 1)) // if moving up or down 2 rows and moving right or left 1 column
	{
		(*invalid_move) = 0;
	} // end of if moving up or down 2 rows and moving right or left 1 column
} // end of KnightMovement function

/* This function checks if bishop can move */
void BishopMovement(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move)
{
	unsigned int i;

	if (abs(move_coordinates[0] - piece_coordinates[0]) == abs(move_coordinates[1] - piece_coordinates[1])) // if diagonal
	{
		if ((move_coordinates[1] == piece_coordinates[1] + 1) || (move_coordinates[1] == piece_coordinates[1] - 1)) // if square just in corner
		{
			(*invalid_move) = 0;
		} // end of if square just in corner
		else if (move_coordinates[1] > piece_coordinates[1] + 1) // if moving up in rows
		{
			if (move_coordinates[0] > piece_coordinates[0] + 1) // if moving right in columns
			{
				(*invalid_move) = 0;
				for (i = 1; i <= abs(move_coordinates[1] - piece_coordinates[1]) - 1; i++) // collision detection
				{
					if (square[piece_coordinates[0] + i][piece_coordinates[1] + i] != 0) // if moving right
					{
						(*invalid_move)++;
					} // end of if moving right
				} // end of collision detection
			} // end of if moving right in columns
			else if (move_coordinates[0] < piece_coordinates[0] - 1) // if moving left in columns
			{
				(*invalid_move) = 0;
				for (i = 1; i <= abs(move_coordinates[0] - piece_coordinates[0]) - 1; i++) // collision detection
				{
					if (square[piece_coordinates[0] - i][piece_coordinates[1] + i] != 0) // if moving left
					{
						(*invalid_move)++;
					} // end of if moving left
				} // end of collision detection
			} // end of if moving left in columns
		} // end of if moving up in rows
		else if (move_coordinates[1] < piece_coordinates[1] - 1) // if moving down in rows
		{
			if (move_coordinates[0] > piece_coordinates[0] + 1) // if moving right in columns
			{
				(*invalid_move) = 0;
				for (i = 1; i <= abs(move_coordinates[0] - piece_coordinates[0]) - 1; i++) // collision detection
				{
					if (square[piece_coordinates[0] + i][piece_coordinates[1] - i] != 0) // if moving right
					{
						(*invalid_move)++;
					} // end of if moving right
				} // end of collision detection
			} // end of if moving right in columns
			else if (move_coordinates[0] < piece_coordinates[0] - 1) // if moving left in columns
			{
				(*invalid_move) = 0;
				for (i = 1; i <= abs(move_coordinates[0] - piece_coordinates[0]) - 1; i++) // collision detection
				{
					if (square[piece_coordinates[0] - i][piece_coordinates[1] - i] != 0) // if moving left
					{
						(*invalid_move)++;
					} // end of if moving left
				} // end of collision detection
			} // end of if moving left in columns
		} // end of if moving down in rows
	} // end of if diagonal
} // end of BishopMovement function

/* This function checks if rook can move */
void RookMovement(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move)
{
	unsigned int i;

	if (move_coordinates[1] == piece_coordinates[1]) // if same row
	{
		if ((move_coordinates[0] == piece_coordinates[0] + 1) || (move_coordinates[0] == piece_coordinates[0] - 1)) // if move left or right a column
		{
			(*invalid_move) = 0;
		} // end of if move left or right a column
		else if (move_coordinates[0] > piece_coordinates[0] + 1) // if moving right more than 1
		{
			(*invalid_move) = 0;
			for (i = 1; i <= abs(move_coordinates[0] - piece_coordinates[0]) - 1; i++) // collision detection
			{
				if (square[piece_coordinates[0] + i][piece_coordinates[1]] != 0) // if squares in betweeen are empty
				{
					(*invalid_move)++;
				} // end of if squares in betweeen are empty
			} // end of collision detection
		} // end of if moving right more than 1
		else if (move_coordinates[0] < piece_coordinates[0] - 1) // if moving left more than 1
		{
			(*invalid_move) = 0;
			for (i = 1; i <= abs(move_coordinates[0] - piece_coordinates[0]) - 1; i++) // collision detection
			{
				if (square[piece_coordinates[0] - i][piece_coordinates[1]] != 0) // if squares in betweeen are empty
				{
					(*invalid_move)++;
				} // end of if squares in betweeen are empty
			} // end of collision detection
		} // end of if moving left more than 1
	} // end of if same row
	else if (move_coordinates[0] == piece_coordinates[0]) // if same column
	{
		if ((move_coordinates[1] == piece_coordinates[1] + 1) || (move_coordinates[1] == piece_coordinates[1] - 1)) // if move up or down a row
		{
			(*invalid_move) = 0;
		} // end of if move up or down a row
		else if (move_coordinates[1] > piece_coordinates[1] + 1) // if moving up more than 1
		{
			(*invalid_move) = 0;
			for (i = 1; i <= abs(move_coordinates[1] - piece_coordinates[1]) - 1; i++) // collision detection
			{
				if (square[piece_coordinates[0]][piece_coordinates[1] + i] != 0) // if squares in betweeen are empty
				{
					(*invalid_move)++;
				} // end of if squares in betweeen are empty
			} // end of collision detection
		} // end of if moving up more than 1
		else if (move_coordinates[1] < piece_coordinates[1] - 1) // if moving down more than 1
		{
			(*invalid_move) = 0;
			for (i = 1; i <= abs(move_coordinates[1] - piece_coordinates[1]) - 1; i++) // collision detection
			{
				if (square[piece_coordinates[0]][piece_coordinates[1] - i] != 0) // if squares in betweeen are empty
				{
					(*invalid_move)++;
				} // end of if squares in betweeen are empty
			} // end of collision detection
		} // end of if moving down more than 1
	} // end of if same column
} // end of RookMovement function

/* This function checks if queen can move */
void QueenMovement(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move)
{
	/*	Diagonals */
	BishopMovement(square, piece_coordinates, move_coordinates, invalid_move);

	/*	Horizontals and verticals */
	RookMovement(square, piece_coordinates, move_coordinates, invalid_move);
} // end of whiteQueen function

/* This function checks if king can move */
void KingMovement(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move)
{
	if (((move_coordinates[0] == piece_coordinates[0] + 1) || (move_coordinates[0] == piece_coordinates[0] - 1)) && (move_coordinates[1] == piece_coordinates[1])) // move U or D a row in same column
	{
		(*invalid_move) = 0;
	} // end of move U or D a row in same column
	else if (((move_coordinates[1] == piece_coordinates[1] + 1) || (move_coordinates[1] == piece_coordinates[1] - 1)) && (move_coordinates[0] == piece_coordinates[0])) // move R or L a column in same row
	{
		(*invalid_move) = 0;
	} // end of move R or L a column in same row
	else if ((move_coordinates[0] == piece_coordinates[0] + 1) && (move_coordinates[1] == piece_coordinates[1] + 1)) // if move up and right
	{
		(*invalid_move) = 0;
	} // end of if move up and right
	else if ((move_coordinates[0] == piece_coordinates[0] + 1) && (move_coordinates[1] == piece_coordinates[1] - 1)) // if move up and left
	{
		(*invalid_move) = 0;
	} // end of if move up and left
	else if ((move_coordinates[0] == piece_coordinates[0] - 1) && (move_coordinates[1] == piece_coordinates[1] + 1)) // if move down and right
	{
		(*invalid_move) = 0;
	} // end of if move down and right
	else if ((move_coordinates[0] == piece_coordinates[0] - 1) && (move_coordinates[1] == piece_coordinates[1] - 1)) // if move down and left
	{
		(*invalid_move) = 0;
	} // end of if move down and left
} // end of KingMovement function

/******************************************************************************************************/
/**************************************** SPECIAL MOVEMENTS *******************************************/
/******************************************************************************************************/

/* This function checks if a white pawn is en passant */
void WhiteEnPassant(int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int human)
{
	if (square[move_coordinates[0]][move_coordinates[1]] == 0) // if square we are moving to isn't occupied
	{
		if (move_coordinates[0] == piece_coordinates[0]) // if moving in same column as piece
		{
			if (piece_coordinates[1] == 1) // if piece in starting row
			{
				if (move_coordinates[1] == piece_coordinates[1] + 2) // if piece is moving up two rows
				{
					if (move_coordinates[0] - 1 >= 0)
					{
						if (square[move_coordinates[0] - 1][move_coordinates[1]] == -1) // if square in the column directly left from where white pawn is moving is a black pawn
						{
							pawn_en_passant_status[0][move_coordinates[0]] = 1;

							if (human == 1)
							{
								printf("White pawn in column %d en passant by black pawn in column %d\n", move_coordinates[0], move_coordinates[0] - 1);
							}
						} // end of if square in the column directly left from where white pawn is moving is a black pawn
					}

					if (move_coordinates[0] + 1 <= 7)
					{
						if (square[move_coordinates[0] + 1][move_coordinates[1]] == -1) // if square in the column directly right from where white pawn is moving is a black pawn
						{
							pawn_en_passant_status[0][move_coordinates[0]] = 1;
							if (human == 1)
							{
								printf("White pawn in column %d en passant by black pawn in column %d\n", move_coordinates[0], move_coordinates[0] + 1);
							}
						} // end of if square in the column directly right from where white pawn is moving is a black pawn
					}
				} // end of if piece is moving up two rows
			} // end of if piece in starting row
		} // end of if moving in same column as piece
	} // end of if square we are moving to isn't occupied
} // end of WhiteEnPassant function

/* This function checks if a black pawn is en passant */
void BlackEnPassant(int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int human)
{
	if (square[move_coordinates[0]][move_coordinates[1]] == 0) // if square we are moving to isn't occupied
	{
		if (move_coordinates[0] == piece_coordinates[0]) // if moving in same column as piece
		{
			if (piece_coordinates[1] == 6) // if piece in starting row
			{
				if (move_coordinates[1] == piece_coordinates[1] - 2) // if piece is moving down two rows
				{
					if (move_coordinates[0] - 1 >= 0)
					{
						if (square[move_coordinates[0] - 1][move_coordinates[1]] == 1) // if square in the column directly left from where black pawn is moving is a white pawn
						{
							pawn_en_passant_status[1][move_coordinates[0]] = 1;
							if (human == 1)
							{
								printf("Black pawn in column %d en passant by white pawn in column %d\n", move_coordinates[0], move_coordinates[0] - 1);
							}
						} // end of if square in the column directly left from where black pawn is moving is a white pawn
					}

					if (move_coordinates[0] + 1 <= 7)
					{
						if (square[move_coordinates[0] + 1][move_coordinates[1]] == 1) // if square in the column directly right from where black pawn is moving is a white pawn
						{
							pawn_en_passant_status[1][move_coordinates[0]] = 1;
							if (human == 1)
							{
								printf("Black pawn in column %d en passant by white pawn in column %d\n", move_coordinates[0], move_coordinates[0] + 1);
							}
						} // end of if square in the column directly right from where white pawn is moving is a black pawn
					}
				} // end of if piece is moving down two rows
			} // end of if piece in starting row
		} // end of if moving in same column as piece
	} // end of if square we are moving to isn't occupied
} // end of BlackEnPassant function

/* This function checks if white queen side castling is valid */
void WhiteQueenSideCastling(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int *invalid_move)
{
	unsigned int i, j, counter;
	int check;

	if (board_state->white_queen_side_castle == 1) // if white queen side castling is still allowed
	{
		if ((piece_coordinates[0] == 4) && (piece_coordinates[1] == 0)) // if piece is white king in original position
		{
			if (square[piece_coordinates[0]][piece_coordinates[1]] == 6) // if square has white king
			{
				int **castle_square;
				castle_square = malloc(sizeof(int*) * chess_constants.number_of_columns);
				for (i = 0; i < chess_constants.number_of_columns; i++)
				{
					castle_square[i] = malloc(sizeof(int) * chess_constants.number_of_rows);
					for (j = 0; j < chess_constants.number_of_rows; j++)
					{
						castle_square[i][j] = square[i][j];
					} // end of j loop
				} // end of i loop

				counter = 3;
				for (i = 1; i <= 3; i++)
				{
					if (square[i][0] == 0) // if squares in between are empty
					{
						counter--; // number of piece in between
					} // end of if squares in between are empty
				} // end of i loop

				if (counter == 0) // if no piece in between
				{
					if (move_coordinates[0] == 2 && move_coordinates[1] == 0) // if moving king to castle position
					{
						int castle_white_king[2];

						check = 0;
						castle_white_king[0] = 4;
						castle_white_king[1] = 0;

						for (i = 3; i >= 1; i--)
						{
							castle_square[i][1] = 6;
							castle_square[i+1][1] = 0;
							castle_white_king[0] = i;
							check = check + InCheckChecker(castle_square, castle_white_king, 1);
						} // end of i loop

						board_state->white_queen_side_castle_allow = 0;
						if (check == 0) // if stayed out of check
						{
							board_state->white_queen_side_castle_allow = 1;
							(*invalid_move) = 0;
						} // end of if stayed out of check
						else
						{
							board_state->white_queen_side_castle_allow = 0;
							(*invalid_move) = 1;
						}
					} // end of if moving king to castle position
				} // end of if no piece in between

				for (i = 0; i < chess_constants.number_of_columns; i++)
				{
					free(castle_square[i]);
				} // end of i loop
				free(castle_square);
			} // end of square has white king
		} // end of if piece is white king in original position
	} // end of if white queen side castling is still allowed
} // end of WhiteQueenSideCastling function

/* This function checks if black queen side castling is valid */
void BlackQueenSideCastling(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int *invalid_move)
{
	unsigned int i, j, counter;
	int check;

	if (board_state->black_queen_side_castle == 1) // if black queen side castling is still allowed
	{
		if ((piece_coordinates[0] == 4) && (piece_coordinates[1] == 7)) // if piece is black king in original position
		{
			if (square[piece_coordinates[0]][piece_coordinates[1]] == -6) // if square has black king
			{
				int **castle_square;
				castle_square = malloc(sizeof(int*) * chess_constants.number_of_columns);
				for (i = 0; i < chess_constants.number_of_columns; i++)
				{
					castle_square[i] = malloc(sizeof(int) * chess_constants.number_of_rows);
					for (j = 0; j < chess_constants.number_of_rows; j++)
					{
						castle_square[i][j] = square[i][j];
					} // end of j loop
				} // end of i loop

				counter = 3;
				for (i = 1; i <= 3; i++)
				{
					if (square[i][7] == 0) // if squares in between are empty
					{
						counter--; // number of piece in between
					} // end of if squares in between are empty
				} // end of i loop

				if (counter == 0) // if no piece in between
				{
					if (move_coordinates[0] == 2 && move_coordinates[1] == 7) // if moving king to castle position
					{
						int castle_black_king[2];

						check = 0;
						castle_black_king[0] = 4;
						castle_black_king[1] = 7;

						for (i = 3; i >= 1; i--)
						{
							castle_square[i][7] = -6;
							castle_square[i+1][7] = 0;
							castle_black_king[0] = i;
							check = check + InCheckChecker(castle_square, castle_black_king, -1);
						} // end of i loop

						board_state->black_queen_side_castle_allow = 0;
						if (check == 0) // if stayed out of check
						{
							board_state->black_queen_side_castle_allow = 1;
							(*invalid_move) = 0;
						} // end of if stayed out of check
						else
						{
							board_state->black_queen_side_castle_allow = 0;
							(*invalid_move) = 1;
						}
					} // end of if moving king to castle position
				} // end of if no piece in between

				for (i = 0; i < chess_constants.number_of_columns; i++)
				{
					free(castle_square[i]);
				} // end of i loop
				free(castle_square);
			} // end of if square has black king
		} // end of if piece is black king in original position
	} // end of if black queen side castling is still allowed
} // end of BlackQueenSideCastling function

/* This function checks if white queen king castling is valid */
void WhiteKingSideCastling(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int *invalid_move)
{
	unsigned int i, j, counter;
	int check;

	if (board_state->white_king_side_castle == 1) // if white king side castling is still allowed
	{
		if ((piece_coordinates[0] == 4) && (piece_coordinates[1] == 0)) // if piece is white king in original position
		{
			if (square[piece_coordinates[0]][piece_coordinates[1]] == 6) // if square has white king
			{
				int **castle_square;
				castle_square = malloc(sizeof(int*) * chess_constants.number_of_columns);
				for (i = 0; i < chess_constants.number_of_columns; i++)
				{
					castle_square[i] = malloc(sizeof(int) * chess_constants.number_of_rows);
					for (j = 0; j < chess_constants.number_of_rows; j++)
					{
						castle_square[i][j] = square[i][j];
					} // end of j loop
				} // end of i loop

				counter = 2;
				for (i = 5; i <= 6; i++)
				{
					if (square[i][0] == 0) // if squares in between are empty
					{
						counter--; // number of piece in between
					} // end of if squares in between are empty
				} // end of i loop

				if (counter == 0) // if no piece in between
				{
					if (move_coordinates[0] == 6 && move_coordinates[1] == 0) // if moving king to castle position
					{
						int castle_white_king[2];

						check = 0;
						castle_white_king[0] = 4;
						castle_white_king[1] = 0;

						for (i = 5; i <= 6; i++)
						{
							castle_square[i][0] = 6;
							castle_square[i-1][0] = 0;
							castle_white_king[0] = i;
							check = check + InCheckChecker(castle_square, castle_white_king, 1);
						} // end of i loop

						board_state->white_king_side_castle_allow = 0;
						if (check == 0) // if stayed out of check
						{
							board_state->white_king_side_castle_allow = 1;
							(*invalid_move) = 0;
						} // end of if stayed out of check
						else
						{
							board_state->white_king_side_castle_allow = 0;
							(*invalid_move) = 1;
						}
					} // end of if moving king to castle position
				} // end of if no piece in between

				for (i = 0; i < chess_constants.number_of_columns; i++)
				{
					free(castle_square[i]);
				} // end of i loop
				free(castle_square);
			} // end of if square has white king
		} // end of if piece is white king in original position
	} // end of if white king side castling is still allowed
} // end of WhiteKingSideCastling function

/* This function checks if black king side castling is valid */
void BlackKingSideCastling(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int *invalid_move)
{
	unsigned int i, j, counter;
	int check;

	if (board_state->black_king_side_castle == 1) // if black king side castling is still allowed
	{
		if ((piece_coordinates[0] == 4) && (piece_coordinates[1] == 7)) // if piece is black king in original position
		{
			if (square[piece_coordinates[0]][piece_coordinates[1]] == -6) // if square has black king
			{
				int **castle_square;
				castle_square = malloc(sizeof(int*) * chess_constants.number_of_columns);
				for (i = 0; i < chess_constants.number_of_columns; i++)
				{
					castle_square[i] = malloc(sizeof(int) * chess_constants.number_of_rows);
					for (j = 0; j < chess_constants.number_of_rows; j++)
					{
						castle_square[i][j] = square[i][j];
					} // end of j loop
				} // end of i loop

				counter = 2;
				for (i = 5; i <= 6; i++)
				{
					if (square[i][7] == 0) // if squares in between are empty
					{
						counter--; // number of piece in between
					} // end of if squares in between are empty
				} // end of i loop

				if (counter == 0) // if no piece in between
				{
					if (move_coordinates[0] == 6 && move_coordinates[1] == 7) // if moving king to castle position
					{
						int castle_black_king[2];

						check = 0;
						castle_black_king[0] = 4;
						castle_black_king[1] = 7;

						for (i = 5; i <= 6; i++)
						{
							castle_square[i][7] = -6;
							castle_square[7][i-1] = 0;
							castle_black_king[0] = i;
							check = check + InCheckChecker(castle_square, castle_black_king, -1);
						} // end of i loop

						board_state->black_king_side_castle_allow = 0;
						if (check == 0) // if stayed out of check
						{
							board_state->black_king_side_castle_allow = 1;
							(*invalid_move) = 0;
						} // end of if stayed out of check
						else
						{
							board_state->black_king_side_castle_allow = 0;
							(*invalid_move) = 1;
						}
					} // end of if moving king to castle position
				} // end of if no piece in between

				for (i = 0; i < chess_constants.number_of_columns; i++)
				{
					free(castle_square[i]);
				} // end of i loop
				free(castle_square);
			} // end of if square has black king
		} // end of if piece is black king in original position
	} // end of if black king side castling is still allowed
} // end of BlackKingSideCastling function

/* This function promotes pawns that reach other side of board */
void PawnPromotion(int **square, int *move_coordinates)
{
	if ((square[move_coordinates[0]][move_coordinates[1]] == 1 && move_coordinates[1] == 7) || (square[move_coordinates[0]][move_coordinates[1]] == -1 && move_coordinates[1] == 0)) // if white or black pawn deserves promotion
	{
		int promote = -9, systemreturn;
		char promote_char, extra_char;

		do
		{
			printf("What do you want to promote your pawn to? 2 = N, 3 = B, 4 = R, 5 = Q\n");

			systemreturn = scanf(" %c", &promote_char);
			if (systemreturn == -1)
			{
				printf("PawnPromotion: Failed reading promote_char\n");
			}
			while ((extra_char = fgetc(stdin)) != '\n' && extra_char != EOF); /* Flush stdin */

			if (promote_char == '2')
			{
				promote = 2;
			}
			else if (promote_char == '3')
			{
				promote = 3;
			}
			else if (promote_char == '4')
			{
				promote = 4;
			}
			else if (promote_char == '5')
			{
				promote = 5;
			}
		} while (promote == -9);

		if (square[move_coordinates[0]][move_coordinates[1]] == 1 && move_coordinates[1] == 7) // if white pawn deserves promotion
		{
			square[move_coordinates[0]][move_coordinates[1]] = promote;
		} // end of if white pawn deserves promotion
		else if (square[move_coordinates[0]][move_coordinates[1]] == -1 && move_coordinates[1] == 0) // black pawn deserves promotion
		{
			square[move_coordinates[0]][move_coordinates[1]] = -promote;
		} // end of if black pawn deserves promotion
	} // end of if white or black pawn deserves promotion
} // end of PawnPromotion function

/* This function disallows castling */
void DisallowCastling(struct BoardState *board_state, int **square)
{
	/* White */
	if (square[4][0] != 6) // if white king is not in original square
	{
		board_state->white_queen_side_castle = 0;
		board_state->white_king_side_castle = 0;
	} // end of if white king is not in original square

	if (square[0][0] != 4) // if white queen side rook is not in original square
	{
		board_state->white_queen_side_castle = 0;
	} // end of if white queen side rook is not in original square

	if (square[7][0] != 4) // if white king side rook is not in original square
	{
		board_state->white_king_side_castle = 0;
	} // end of if white king side rook is not in original square

	/* Black */
	if (square[4][7] != -6) // if black king is not in original square
	{
		board_state->black_queen_side_castle = 0;
		board_state->black_king_side_castle = 0;
	} // end of if black king is not in original square

	if (square[0][7] != -4) // if black queen side rook is not in original square
	{
		board_state->black_queen_side_castle = 0;
	} // end of if black queen side rook is not in original square

	if (square[7][7] != -4) // if black king side rook is not in original square
	{
		board_state->black_king_side_castle = 0;
	} // end of if black king side rook is not in original square
} // end of DisallowCastling function

/******************************************************************************************************/
/***************************************** CHECK FOR CHECK ********************************************/
/******************************************************************************************************/

/* This function checks valid moves for check */
int CheckValidMovesForCheck(int *piece_coordinates, int *move_coordinates, int **square, int ****piece_positions)
{
	unsigned int i, j;
	int check;

	int **check_square;
	check_square = malloc(sizeof(int*) * chess_constants.number_of_columns);
	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		check_square[i] = malloc(sizeof(int) * chess_constants.number_of_rows);
		for (j = 0; j < chess_constants.number_of_rows; j++)
		{
			check_square[i][j] = square[i][j];
		} // end of j loop
	} // end of i loop

	check_square[move_coordinates[0]][move_coordinates[1]] = check_square[piece_coordinates[0]][piece_coordinates[1]]; // move piece to new spot
	check_square[piece_coordinates[0]][piece_coordinates[1]] = 0; // remove piece from old spot

	if (check_square[move_coordinates[0]][move_coordinates[1]] > 0) // if moved piece is white
	{
		int king_coordinates[2];

		if (check_square[move_coordinates[0]][move_coordinates[1]] == 6) // if we are moving white king
		{
			for (i = 0; i < 2; i++)
			{
				king_coordinates[i] = move_coordinates[i];
			} // end of i loop
		} // end of if we are moving white king
		else // if we are NOT moving white king
		{
			for (i = 0; i < 2; i++)
			{
				king_coordinates[i] = piece_positions[0][5][0][i];
			} // end of i loop
		} // end of if we are NOT moving white king

		check = InCheckChecker(check_square, king_coordinates, 1);
	} // end of if moved piece is white
	else if (check_square[move_coordinates[0]][move_coordinates[1]] < 0) // if moved piece is black
	{
		int king_coordinates[2];

		if (check_square[move_coordinates[0]][move_coordinates[1]] == -6) // if we are moving black king
		{
			for (i = 0; i < 2; i++)
			{
				king_coordinates[i] = move_coordinates[i];
			} // end of i loop
		} // end of if we are moving black king
		else // if we are NOT moving black king
		{
			for (i = 0; i < 2; i++)
			{
				king_coordinates[i] = piece_positions[1][5][0][i];
			} // end of i loop
		} // end of if we are NOT moving black king

		check = InCheckChecker(check_square, king_coordinates, -1);
	} // end of if moved piece is black

	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		free(check_square[i]);
	} // end of i loop
	free(check_square);

	return check;
} // end of CheckValidMovesForCheck function

/* This function checks if in check */
int InCheckChecker(int **square, int *king_coordinates, int check_color)
{
	int in_check = 0;

	/* Pawns */
	if (check_color == 1)
	{
		in_check = WhiteInCheckDueToPawns(square, king_coordinates);
	}
	else
	{
		in_check = BlackInCheckDueToPawns(square, king_coordinates);
	}

	/* Knights */
	if (in_check == 0) // if white still not in check
	{
		in_check = InCheckDueToKnights(square, king_coordinates, check_color);

		/* Horizontals */
		if (in_check == 0) // if white still not in check
		{
			in_check = InCheckDueToHorizontals(square, king_coordinates, check_color);

			/* Verticals */
			if (in_check == 0) // if white still not in check
			{
				in_check = InCheckDueToVerticals(square, king_coordinates, check_color);

				/* Diagonals */
				if (in_check == 0) // if white still not in check
				{
					in_check = InCheckDueToDiagonals(square, king_coordinates, check_color);
				} // end of if white still not in check
			} // end of if white still not in check
		} // end of if white still not in check
	} // end of if white still not in check

	return in_check;
} // end of function InCheckChecker

/* This function checks if white is in check due to pawns */
int WhiteInCheckDueToPawns(int **square, int *king_coordinates)
{
	int white_in_check = 0;

	if ((king_coordinates[0] - 1 >= 0 && king_coordinates[1] + 1 <= 7 && square[king_coordinates[0] - 1][king_coordinates[1] + 1] == -1) || (king_coordinates[0] + 1 <= 7 && king_coordinates[1] + 1 <= 7 && square[king_coordinates[0] + 1][king_coordinates[1] + 1] == -1))
	{
		white_in_check = 1;
	}

	return white_in_check;
} // end of WhiteInCheckDueToPawns function

/* This function checks if black is in check due to pawns */
int BlackInCheckDueToPawns(int **square, int *king_coordinates)
{
	int black_in_check = 0;

	if ((king_coordinates[0] - 1 >= 0 && king_coordinates[1] - 1 >= 0 && square[king_coordinates[0] - 1][king_coordinates[1] - 1] == 1) || (king_coordinates[0] + 1 <= 7 && king_coordinates[1] - 1 >= 0 && square[king_coordinates[0] + 1][king_coordinates[1] - 1] == 1))
	{
		black_in_check = 1;
	}

	return black_in_check;
} // end of BlackInCheckDueToPawns function

/* This function checks if in check due to knights */
int InCheckDueToKnights(int **square, int *king_coordinates, int check_color)
{
	int in_check = 0;

	if (king_coordinates[0] + 1 <= 7 && king_coordinates[1] + 2 <= 7 && square[king_coordinates[0] + 1][king_coordinates[1] + 2] == -2 * check_color) // if right 1 column, up 2 rows
	{
		in_check = 1;
	}
	else if (king_coordinates[0] + 1 <= 7 && king_coordinates[1] - 2 >= 0 && square[king_coordinates[0] + 1][king_coordinates[1] - 2] == -2 * check_color) // if right 1 column, down 2 rows
	{
		in_check = 1;
	}
	else if (king_coordinates[0] - 1 >= 0 && king_coordinates[1] + 2 <= 7 && square[king_coordinates[0] - 1][king_coordinates[1] + 2] == -2 * check_color) // if left 1 column, up 2 rows
	{
		in_check = 1;
	}
	else if (king_coordinates[0] - 1 >= 0 && king_coordinates[1] - 2 >= 0 && square[king_coordinates[0] - 1][king_coordinates[1] - 2] == -2 * check_color) // if left 1 column, down 2 rows
	{
		in_check = 1;
	}
	else if (king_coordinates[0] + 2 <= 7 && king_coordinates[1] + 1 <= 7 && square[king_coordinates[0] + 2][king_coordinates[1] + 1] == -2 * check_color) // if right 2 columns, up 1 row
	{
		in_check = 1;
	}
	else if (king_coordinates[0] + 2 <= 7 && king_coordinates[1] - 1 >= 0 && square[king_coordinates[0] + 2][king_coordinates[1] - 1] == -2 * check_color) // if right 2 columns, down 1 row
	{
		in_check = 1;
	}
	else if (king_coordinates[0] - 2 >= 0 && king_coordinates[1] + 1 <= 7 && square[king_coordinates[0] - 2][king_coordinates[1] + 1] == -2 * check_color) // if left 2 columns, up 1 row
	{
		in_check = 1;
	}
	else if (king_coordinates[0] - 2 >= 0 && king_coordinates[1] - 1 >= 0 && square[king_coordinates[0] - 2][king_coordinates[1] - 1] == -2 * check_color) // if left 2 columns, down 1 row
	{
		in_check = 1;
	}

	return in_check;
} // end of InCheckDueToKnights function

/* This function checks if in check due to horizontals */
int InCheckDueToHorizontals(int **square, int *king_coordinates, int check_color)
{
	int i, j, in_check = 0;

	for (i = king_coordinates[0] + 1; i < chess_constants.number_of_columns; i++) // move right in columns in row of white king
	{
		if (square[i][king_coordinates[1]] * check_color > 0) // if square is occupied by white
		{
			break; // blocked from being in check so can break out of loop
		}
		else if (square[i][king_coordinates[1]] * check_color < 0) // if square is occupied by black
		{
			if (square[i][king_coordinates[1]] == -4 * check_color || square[i][king_coordinates[1]] == -5 * check_color) // if square is occupied by black rook or black queen
			{
				in_check = 1;
				break;
			} // end of if square is occupied by black rook or black queen
			else // if square is NOT occupied by black rook or black queen
			{
				if (i == king_coordinates[0] + 1) // if square is just to the right of white king
				{
					if (square[i][king_coordinates[1]] == -6 * check_color) // if square is occupied by black king
					{
						in_check = 1;
						break;
					} // end of if square is occupied by black king
					else // if square is NOT occupied by black king
					{
						break;
					} // end of if square is NOT occupied by black king
				} // end of if square is just to the right of white king
				else // if square is NOT just to the right of white king
				{
					break;
				} // end of if square is NOT just to the right of white king
			} // end of if square is NOT occupied by black rook or black queen
		} // end of if square is occupied by black
	} // end i loop

	if (in_check == 0) // if white still not in check
	{
		for (i = king_coordinates[0] - 1; i >= 0; i--) // move left in columns in row of white king
		{
			if (square[i][king_coordinates[1]] * check_color > 0) // if square is occupied by white
			{
				break; // blocked from being in check so can break out of loop
			}
			else if (square[i][king_coordinates[1]] * check_color < 0) // if square is occupied by black
			{
				if (square[i][king_coordinates[1]] == -4 * check_color || square[i][king_coordinates[1]] == -5 * check_color) // if square is occupied by black rook or black queen
				{
					in_check = 1;
					break;
				} // end of if square is occupied by black rook or black queen
				else // if square is NOT occupied by black rook or black queen
				{
					if (i == king_coordinates[0] - 1) // if square is just to the left of white king
					{
						if (square[i][king_coordinates[1]] == -6 * check_color) // if square is occupied by black king
						{
							in_check = 1;
							break;
						} // end of if square is occupied by black king
						else // if square is NOT occupied by black king
						{
							break;
						} // end of if square is NOT occupied by black king
					} // end of if square is just to the left of white king
					else // if square is NOT just to the left of white king
					{
						break;
					} // end of if square is NOT just to the left of white king
				} // end of if square is NOT occupied by black rook or black queen
			} // end of if square is occupied by black
		} // end i loop
	} // end of if white is still not in check

	return in_check;
} // end of InCheckDueToHorizontals function

/* This function checks if in check due to verticals */
int InCheckDueToVerticals(int **square, int *king_coordinates, int check_color)
{
	int i, j, in_check = 0;

	for (i = king_coordinates[1] + 1; i < chess_constants.number_of_rows; i++) // move up in rows in column of white king
	{
		if (square[king_coordinates[0]][i] * check_color > 0) // if square is occupied by white
		{
			break; // blocked from being in check so can break out of loop
		}
		else if (square[king_coordinates[0]][i] * check_color < 0) // if square is occupied by black
		{
			if (square[king_coordinates[0]][i] == -4 * check_color || square[king_coordinates[0]][i] == -5 * check_color) // if square is occupied by black rook or black queen
			{
				in_check = 1;
				break;
			} // end of if square is occupied by black rook or black queen
			else // if square is NOT occupied by black rook or black queen
			{
				if (i == king_coordinates[1] + 1) // if square is just above white king
				{
					if (square[king_coordinates[0]][i] == -6 * check_color) // if square is occupied by black king
					{
						in_check = 1;
						break;
					} // end of if square is occupied by black king
					else // if square is NOT occupied by black king
					{
						break;
					} // end of if square is NOT occupied by black king
				} // end of if square is just above white king
				else // if square is NOT just above white king
				{
					break;
				} // end of if square is NOT just above white king
			} // end of if square is NOT occupied by black rook or black queen
		} // end of if square is occupied by black
	} // end i loop

	if (in_check == 0) // if white still not in check
	{
		for (i = king_coordinates[1] - 1; i >= 0; i--) // move down in rows in column of white king
		{
			if (square[king_coordinates[0]][i] * check_color > 0) // if square is occupied by white
			{
				break; // blocked from being in check so can break out of loop
			}
			else if (square[king_coordinates[0]][i] * check_color < 0) // if square is occupied by black
			{
				if (square[king_coordinates[0]][i] == -4 * check_color || square[king_coordinates[0]][i] == -5 * check_color) // if square is occupied by black rook or black queen
				{
					in_check = 1;
					break;
				} // end of if square is occupied by black rook or black queen
				else // if square is NOT occupied by black rook or black queen
				{
					if (i == king_coordinates[1] - 1) // if square is just below white king
					{
						if (square[king_coordinates[0]][i] == -6 * check_color) // if square is occupied by black king
						{
							in_check = 1;
							break;
						} // end of if square is occupied by black king
						else // if square is NOT occupied by black king
						{
							break;
						} // end of if square is NOT occupied by black king
					} // end of if square is just below white king
					else // if square is NOT just below white king
					{
						break;
					} // end of if square is NOT just below white king
				} // end of if square is NOT occupied by black rook or black queen
			} // end of if square is occupied by black
		} // end i loop
	} // end of if white is still not in check

	return in_check;
} // end of InCheckDueToVerticals function

/* This function checks if in check due to diagonals */
int InCheckDueToDiagonals(int **square, int *king_coordinates, int check_color)
{
	int i, j, in_check = 0;

	for (i = king_coordinates[0] + 1; i < chess_constants.number_of_columns; i++) // move up and right from white king
	{
		j = i - king_coordinates[0];

		if (king_coordinates[1] + j >= 0 && king_coordinates[1] + j <= 7)
		{
			if (square[i][king_coordinates[1] + j] * check_color > 0) // if square is occupied by white
			{
				break; // blocked from being in check so can break out of loop
			}
			else if (square[i][king_coordinates[1] + j] * check_color < 0) // if square is occupied by black
			{
				if (square[i][king_coordinates[1] + j] == -3 * check_color || square[i][king_coordinates[1] + j] == -5 * check_color) // if square is occupied by black bishop or black queen
				{
					in_check = 1;
					break;
				} // end of if square is occupied by black rook or black queen
				else // if square is NOT occupied by black bishop or black queen
				{
					if (i == king_coordinates[0] + 1) // if square is just to the right of white king
					{
						if (square[i][king_coordinates[1] + j] == -6 * check_color) // if square is occupied by black king
						{
							in_check = 1;
							break;
						} // end of if square is occupied by black king
						else // if square is NOT occupied by black king
						{
							break;
						} // end of if square is NOT occupied by black king
					} // end of if square is just to the right of white king
					else // if square is NOT just to the right of white king
					{
						break;
					} // end of if square is NOT just to the right of white king
				} // end of if square is NOT occupied by black bishop or black queen
			} // end of if square is occupied by black
		}
	} // end i loop

	if (in_check == 0) // if white still not in check
	{
		for (i = king_coordinates[0] - 1; i >= 0; i--) // move up and left from white king
		{
			j = i - king_coordinates[0];

			if (king_coordinates[1] - j >= 0 && king_coordinates[1] - j <= 7)
			{
				if (square[i][king_coordinates[1] - j] * check_color > 0) // if square is occupied by white
				{
					break; // blocked from being in check so can break out of loop
				}
				else if (square[i][king_coordinates[1] - j] * check_color < 0) // if square is occupied by black
				{
					if (square[i][king_coordinates[1] - j] == -3 * check_color || square[i][king_coordinates[1] - j] == -5 * check_color) // if square is occupied by black bishop or black queen
					{
						in_check = 1;
						break;
					} // end of if square is occupied by black rook or black queen
					else // if square is NOT occupied by black bishop or black queen
					{
						if (i == king_coordinates[0] - 1) // if square is just to the left of white king
						{
							if (square[i][king_coordinates[1] - j] == -6 * check_color) // if square is occupied by black king
							{
								in_check = 1;
								break;
							} // end of if square is occupied by black king
							else // if square is NOT occupied by black king
							{
								break;
							} // end of if square is NOT occupied by black king
						} // end of if square is just to the left of white king
						else // if square is NOT just to the left of white king
						{
							break;
						} // end of if square is NOT just to the left of white king
					} // end of if square is NOT occupied by black bishop or black queen
				} // end of if square is occupied by black
			}
		} // end i loop

		if (in_check == 0) // if white still not in check
		{
			for (i = king_coordinates[0] + 1; i < chess_constants.number_of_columns; i++) // move down and right from white king
			{
				j = i - king_coordinates[0];

				if (king_coordinates[1] - j >= 0 && king_coordinates[1] - j <= 7)
				{
					if (square[i][king_coordinates[1] - j] * check_color > 0) // if square is occupied by white
					{
						break; // blocked from being in check so can break out of loop
					}
					else if (square[i][king_coordinates[1] - j] * check_color < 0) // if square is occupied by black
					{
						if (square[i][king_coordinates[1] - j] == -3 * check_color || square[i][king_coordinates[1] - j] == -5 * check_color) // if square is occupied by black bishop or black queen
						{
							in_check = 1;
							break;
						} // end of if square is occupied by black rook or black queen
						else // if square is NOT occupied by black bishop or black queen
						{
							if (i == king_coordinates[0] + 1) // if square is just to the right of white king
							{
								if (square[i][king_coordinates[1] - j] == -6 * check_color) // if square is occupied by black king
								{
									in_check = 1;
									break;
								} // end of if square is occupied by black king
								else // if square is NOT occupied by black king
								{
									break;
								} // end of if square is NOT occupied by black king
							} // end of if square is just to the right of white king
							else // if square is NOT just to the right of white king
							{
								break;
							} // end of if square is NOT just to the right of white king
						} // end of if square is NOT occupied by black bishop or black queen
					} // end of if square is occupied by black
				}
			} // end i loop

			if (in_check == 0) // if white still not in check
			{
				for (i = king_coordinates[0] - 1; i >= 0; i--) // move down and left from white king
				{
					j = i - king_coordinates[0];

					if (king_coordinates[1] + j >= 0 && king_coordinates[1] + j <= 7)
					{
						if (square[i][king_coordinates[1] + j] * check_color > 0) // if square is occupied by white
						{
							break; // blocked from being in check so can break out of loop
						}
						else if (square[i][king_coordinates[1] + j] * check_color < 0) // if square is occupied by black
						{
							if (square[i][king_coordinates[1] + j] == -3 * check_color || square[i][king_coordinates[1] + j] == -5 * check_color) // if square is occupied by black bishop or black queen
							{
								in_check = 1;
								break;
							} // end of if square is occupied by black rook or black queen
							else // if square is NOT occupied by black bishop or black queen
							{
								if (i == king_coordinates[0] - 1) // if square is just to the left of white king
								{
									if (square[i][king_coordinates[1] + j] == -6 * check_color) // if square is occupied by black king
									{
										in_check = 1;
										break;
									} // end of if square is occupied by black king
									else // if square is NOT occupied by black king
									{
										break;
									} // end of if square is NOT occupied by black king
								} // end of if square is just to the left of white king
								else // if square is NOT just to the left of white king
								{
									break;
								} // end of if square is NOT just to the left of white king
							} // end of if square is NOT occupied by black bishop or black queen
						} // end of if square is occupied by black
					}
				} // end i loop
			} // end of if white is still not in check
		} // end of if white is still not in check
	} // end of if white is still not in check

	return in_check;
} // end of WhiteInCheckDueToDiagonals function

/******************************************************************************************************/
/****************************************** PERFORM MOVES *********************************************/
/******************************************************************************************************/

/* This function performs the selected human moves */
int PerformHumanMoves(struct PlayerOptions *player_options, struct BoardState *board_state, char *basic_chessboard, int turn, int **square, int **pawn_en_passant_status, unsigned int **piece_count, int *piece_coordinates, int *move_coordinates, int ****piece_positions, int **color, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history)
{
	if (board_state->turn % 2 == 1) // if it is white's turn
	{
		if (square[piece_coordinates[0]][piece_coordinates[1]] == 1) // if piece we moved was a white pawn
		{
			WhiteEnPassant(square, piece_coordinates, move_coordinates, pawn_en_passant_status, 1);
		} // end of if piece we moved was a white pawn
	} // end of if it is white's turn
	else // if it is black's turn
	{
		if (square[piece_coordinates[0]][piece_coordinates[1]] == -1) // if piece we moved was a black pawn
		{
			BlackEnPassant(square, piece_coordinates, move_coordinates, pawn_en_passant_status, 1);
		} // end of if piece we moved was a black pawn
	} // end of if it is black's turn

	PerformValidatedMoves(board_state, square, piece_coordinates, move_coordinates, piece_positions, 1);

	PawnPromotion(square, move_coordinates);

	DisallowCastling(board_state, square);

	CountPieces(board_state, square, piece_count, piece_positions);

	if (board_state->turn % 2 == 1) // if it is white's turn
	{
		board_state->white_in_check = 0;
		board_state->black_in_check = InCheckChecker(square, piece_positions[1][5][0], -1);

		if (board_state->black_in_check == 1) // if black is now in check
		{
			printf("Black is now in check!\n");
		} // end of if black is now in check
	} // end of if it is white's turn
	else // if it is black's turn
	{
		board_state->black_in_check = 0;
		board_state->white_in_check = InCheckChecker(square, piece_positions[0][5][0], 1);

		if (board_state->white_in_check == 1) // if white is now in check
		{
			printf("White is now in check!\n");
		} // end of if white is now in check
	} // end of if it is black's turn

	PrintPieces(board_state->turn, square, color, player_options->players, basic_chessboard);

	if (CheckIfGameOver(board_state, board_state->turn, square, pawn_en_passant_status, piece_count, move_coordinates, piece_positions, color, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, 1) == 1)
	{
		return board_state->game_over;
	}

	/******************************************************************************************************/
	/********************************* CALCULATE BOARD EVALUATION SCORE ***********************************/
	/******************************************************************************************************/

	board_state->evaluation_score = BoardEvaluationScore(board_state, board_state->turn, square, color, piece_positions, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers);

	PrintPositionGamePhaseScore(board_state);

	return board_state->game_over;
} // end of PerformHumanMoves function

/* This function performs validated moves */
void PerformValidatedMoves(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int ****piece_positions, int human)
{
	square[move_coordinates[0]][move_coordinates[1]] = square[piece_coordinates[0]][piece_coordinates[1]]; // move piece to new square
	square[piece_coordinates[0]][piece_coordinates[1]] = 0; // remove piece from old square

	/* Remove captured pawn from en passant */
	if (square[move_coordinates[0]][move_coordinates[1]] == 1) // if piece moved was white pawn
	{
		if (board_state->en_passant_captured != -9)
		{
			square[move_coordinates[0]][move_coordinates[1] - 1] = 0;
		}
	} // end of if piece moved was white pawn
	else if (square[move_coordinates[0]][move_coordinates[1]] == -1) // if piece moved was black pawn
	{
		if (board_state->en_passant_captured != -9)
		{
			square[move_coordinates[0]][move_coordinates[1] + 1] = 0;
		}
	} // end of if piece moved was black pawn

	/* If castling, move rook too */
	if (board_state->white_queen_side_castle_allow == 1 && board_state->white_queen_side_castle == 1)
	{
		square[3][0] = 4;
		square[0][0] = 0;
		if (human == 1)
		{
			printf("White castled queen side!\n");
		}
	}
	else if (board_state->black_queen_side_castle_allow == 1 && board_state->black_queen_side_castle == 1)
	{
		square[3][7] = -4;
		square[0][7] = 0;
		if (human == 1)
		{
			printf("Black castled queen side!\n");
		}
	}
	else if (board_state->white_king_side_castle_allow == 1 && board_state->white_king_side_castle == 1)
	{
		square[5][0] = 4;
		square[7][0] = 0;
		if (human == 1)
		{
			printf("White castled king side!\n");
		}
	}
	else if (board_state->black_king_side_castle_allow == 1 && board_state->black_king_side_castle == 1)
	{
		square[5][7] = -4;
		square[7][7] = 0;
		if (human == 1)
		{
			printf("Black castled king side!\n");
		}
	}
} // end of PerformValidatedMoves function

/******************************************************************************************************/
/*************************************** CHECK FOR GAME OVER ******************************************/
/******************************************************************************************************/

/* This function checks if the game is over according to the rules in its current state */
int CheckIfGameOver(struct BoardState *board_state, int turn, int **square, int **pawn_en_passant_status, unsigned int **piece_count, int *move_coordinates, int ****piece_positions, int **color, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int human)
{
	//------------------------------------------------------------Check for Checkmate or Stalemate------------------------------------------------------------

	board_state->game_over = NoLegalMovesMateChecker(board_state, board_state->turn, square, pawn_en_passant_status, piece_count, piece_positions, color, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, human);
	if (board_state->game_over == 1) // if there is a mate
	{
		if (board_state->turn % 2 == 1) // if it is white's turn
		{
			if (board_state->black_in_check == 1) // if black is in check
			{
				printf("White checkmates black! White has won the game!\n");
			} // end of if black is in check
			else // if black is NOT in check
			{
				printf("No valid moves! Stalemate! It is a draw!\n");
			} // end of if black is NOT in check
		} // end of if it is white's turn
		else // if it is black's turn
		{
			if (board_state->white_in_check == 1) // if white is in check
			{
				printf("Black checkmates white! Black has won the game!\n");
			} // end of if white is in check
			else // if white is NOT in check
			{
				printf("No valid moves! Stalemate! It is a draw!\n");
			} // end of if white is NOT in check
		} // end of if it is black's turn

		return board_state->game_over;
	} // end of if there is a mate

	//-----------------------------------------------------------------Check for Threefold Repetition----------------------------------------------------------------

	board_state->game_over = ThreeFoldRepetition(board_state->turn, square, board_history);
	if (board_state->game_over == 1) // if there has been three fold repetition
	{
		printf("The same board state has happened three times! It is a draw!\n");

		return board_state->game_over;
	} // end of if there has been three fold repetition

	//-------------------------------------------------------------------Check for Fifty-move Rule-------------------------------------------------------------------

	board_state->game_over = FiftyMoveRule(board_state, move_coordinates, square);
	if (board_state->game_over == 1) // if there have been fifty moves without a pawn move AND piece capture
	{
		printf("There hasn't been a pawn move AND piece capture in 50 turns! It is a draw!\n");

		return board_state->game_over;
	} // end of if there have been fifty moves without a pawn move AND piece capture

	//---------------------------------------------------------------Check for Checkmate Impossibility---------------------------------------------------------------

	board_state->game_over = CheckmateImpossibility(piece_count, square, color);
	if (board_state->game_over == 1) // if checkmate is impossible
	{
		printf("Checkmate is impossible! It is a draw!\n");

		return board_state->game_over;
	} // end of if checkmate is impossible

	return board_state->game_over;
} // end of CheckIfGameOver function

/* This function checks for checkmate and stalemate */
int NoLegalMovesMateChecker(struct BoardState *board_state, int turn, int **square, int **pawn_en_passant_status, unsigned int **piece_count, int ****piece_positions, int **color, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int human)
{
	unsigned int i, j, moves_tried = 0, moves_valid = 0;
	int invalid_move = 1, mated;
	int depth = -9, piece_number = -9, old_piece_square = -9, old_move_square = -9;
	int best_score = -9, best_piece_file = -9, best_piece_rank = -9, best_move_file = -9, best_move_rank = -9;

	struct BoardState old_board_state;
	old_board_state = (*board_state);

	struct BoardState best_board_state;
	InitializeBoardStateVariables(&best_board_state);

	int piece_coordinates[2];
	int move_coordinates[2];

	if (turn % 2 == 0) // if it is black's turn
	{
		for (i = 0; i < piece_count[0][0]; i++)
		{
			piece_coordinates[0] = piece_positions[0][0][i][0];
			piece_coordinates[1] = piece_positions[0][0][i][1];

			TryWhitePawnMoves(board_state, piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
		} // end of i loop

		for (i = 0; i < piece_count[0][1]; i++)
		{
			piece_coordinates[0] = piece_positions[0][1][i][0];
			piece_coordinates[1] = piece_positions[0][1][i][1];

			TryKnightMoves(piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
		} // end of i loop

		for (i = 0; i < piece_count[0][2]; i++)
		{
			piece_coordinates[0] = piece_positions[0][2][i][0];
			piece_coordinates[1] = piece_positions[0][2][i][1];

			TryBishopMoves(piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
		} // end of i loop

		for (i = 0; i < piece_count[0][3]; i++)
		{
			piece_coordinates[0] = piece_positions[0][3][i][0];
			piece_coordinates[1] = piece_positions[0][3][i][1];

			TryRookMoves(piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
		} // end of i loop

		for (i = 0; i < piece_count[0][4]; i++)
		{
			piece_coordinates[0] = piece_positions[0][4][i][0];
			piece_coordinates[1] = piece_positions[0][4][i][1];

			TryQueenMoves(piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
		} // end of i loop

		piece_coordinates[0] = piece_positions[0][5][0][0];
		piece_coordinates[1] = piece_positions[0][5][0][1];

		TryKingMoves(board_state, piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
	} // end of if it is black's turn
	else // if it is white's turn
	{
		for (i = 0; i < piece_count[1][0]; i++)
		{
			piece_coordinates[0] = piece_positions[1][0][i][0];
			piece_coordinates[1] = piece_positions[1][0][i][1];

			TryBlackPawnMoves(board_state, piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
		} // end of i loop

		for (i = 0; i < piece_count[1][1]; i++)
		{
			piece_coordinates[0] = piece_positions[1][1][i][0];
			piece_coordinates[1] = piece_positions[1][1][i][1];

			TryKnightMoves(piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
		} // end of i loop

		for (i = 0; i < piece_count[1][2]; i++)
		{
			piece_coordinates[0] = piece_positions[1][2][i][0];
			piece_coordinates[1] = piece_positions[1][2][i][1];

			TryBishopMoves(piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
		} // end of i loop

		for (i = 0; i < piece_count[1][3]; i++)
		{
			piece_coordinates[0] = piece_positions[1][3][i][0];
			piece_coordinates[1] = piece_positions[1][3][i][1];

			TryRookMoves(piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
		} // end of i loop

		for (i = 0; i < piece_count[1][4]; i++)
		{
			piece_coordinates[0] = piece_positions[1][4][i][0];
			piece_coordinates[1] = piece_positions[1][4][i][1];

			TryQueenMoves(piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
		} // end of i loop

		piece_coordinates[0] = piece_positions[1][5][0][0];
		piece_coordinates[1] = piece_positions[1][5][0][1];

		TryKingMoves(board_state, piece_coordinates, move_coordinates, square, pawn_en_passant_status, &invalid_move, &moves_tried, &moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0, turn, depth, piece_number, &old_piece_square, &old_move_square, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_board_state, &old_board_state);
	} // end of if it is white's turn

	if (human == 1)
	{
		if (turn % 2 == 0) // if it is black's turn
		{
			printf("Tried all white pieces, moves_tried = %d, moves_valid = %d\n", moves_tried, moves_valid);
		} // end of if it is black's turn
		else // if it is white's turn
		{
			printf("Tried all black pieces, moves_tried = %d, moves_valid = %d\n", moves_tried, moves_valid);
		} // end of if it is white's turn
	}

	if (moves_valid == 0) // if there are no valid moves
	{
		mated = 1;
	} // end of if there are no valid moves
	else // if there are valid moves
	{
		mated = 0;

	} // end of if there are valid moves

	return mated;
} // end of NoLegalMovesMateChecker function

/* This function checks move validity for check */
void CheckHumanMoveValidityForCheck(int *piece_coordinates, int *move_coordinates, int **square, int ****piece_positions, int *invalid_move, unsigned int *moves_valid)
{
	if ((*invalid_move) == 0) // if move is valid so far
	{
		(*invalid_move) = CheckValidMovesForCheck(piece_coordinates, move_coordinates, square, piece_positions);

		if ((*invalid_move) == 0) // if move is still valid after checking for check
		{
			(*moves_valid)++;
		} // end of if move is still valid after checking for check
	} // end of if move is valid so far

} // end of CheckHumanMoveValidityForCheck function

/******************************************************************************************************/
/******************************************** TRY MOVES ***********************************************/
/******************************************************************************************************/

/* This function tries all white pawn moves */
void TryWhitePawnMoves(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state)
{
	/* Move pawn up 2 */
	if (piece_coordinates[1] + 2 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = piece_coordinates[1] + 2;

		(*moves_tried)++;

		(*invalid_move) = 1;
		if (computer == 1) // if this is a computer player
		{
			ApplyWhitePawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 0);
		} // end of if this is a computer player
		else // if this is NOT a computer player
		{
			ApplyWhitePawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 1);
		} // end of if this is NOT a computer player

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move pawn up 1 */
	if (piece_coordinates[1] + 1 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = piece_coordinates[1] + 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		if (computer == 1) // if this is a computer player
		{
			ApplyWhitePawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 0);
		} // end of if this is a computer player
		else // if this is NOT a computer player
		{
			ApplyWhitePawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 1);
		} // end of if this is NOT a computer player

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move pawn to capture left */
	if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] + 1 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] - 1;
		move_coordinates[1] = piece_coordinates[1] + 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		if (computer == 1) // if this is a computer player
		{
			ApplyWhitePawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 0);
		} // end of if this is a computer player
		else // if this is NOT a computer player
		{
			ApplyWhitePawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 1);
		} // end of if this is NOT a computer player

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move pawn to capture right */
	if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] + 1 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] + 1;
		move_coordinates[1] = piece_coordinates[1] + 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		if (computer == 1) // if this is a computer player
		{
			ApplyWhitePawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 0);
		} // end of if this is a computer player
		else // if this is NOT a computer player
		{
			ApplyWhitePawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 1);
		} // end of if this is NOT a computer player

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

} // end of TryWhitePawnMoves function

/* This function applies white pawn moves */
void ApplyWhitePawnMoves(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int *invalid_move, int mate_checking)
{
	if ((square[move_coordinates[0]][move_coordinates[1]] <= 0) && (square[move_coordinates[0]][move_coordinates[1]] != -6)) // if move square isn't white or black king
	{
		WhitePawnMovement(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, mate_checking);
	} // end of if move square isn't white or black king
} // end of ApplyWhitePawnMoves function

/* This function tries all black pawn moves */
void TryBlackPawnMoves(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state)
{
	/* Move pawn down 2 */
	if (piece_coordinates[1] - 2 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = piece_coordinates[1] - 2;

		(*moves_tried)++;

		(*invalid_move) = 1;
		if (computer == 1) // if this is a computer player
		{
			ApplyBlackPawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 0);
		} // end of if this is a computer player
		else // if this is NOT a computer player
		{
			ApplyBlackPawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 1);
		} // end of if this is NOT a computer player

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal

	} // end of if still on board

	/* Move pawn down 1 */
	if (piece_coordinates[1] - 1 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = piece_coordinates[1] - 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		if (computer == 1) // if this is a computer player
		{
			ApplyBlackPawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 0);
		} // end of if this is a computer player
		else // if this is NOT a computer player
		{
			ApplyBlackPawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 1);
		} // end of if this is NOT a computer player

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move pawn to capture left */
	if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] - 1 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] - 1;
		move_coordinates[1] = piece_coordinates[1] - 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		if (computer == 1) // if this is a computer player
		{
			ApplyBlackPawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 0);
		} // end of if this is a computer player
		else // if this is NOT a computer player
		{
			ApplyBlackPawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 1);
		} // end of if this is NOT a computer player

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move pawn to capture right */
	if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] - 1 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] + 1;
		move_coordinates[1] = piece_coordinates[1] - 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		if (computer == 1) // if this is a computer player
		{
			ApplyBlackPawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 0);
		} // end of if this is a computer player
		else // if this is NOT a computer player
		{
			ApplyBlackPawnMoves(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, 1);
		} // end of if this is NOT a computer player

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

} // end of TryBlackPawnMoves function

/* This function applies black pawn moves */
void ApplyBlackPawnMoves(struct BoardState *board_state, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int *invalid_move, int mate_checking)
{
	if ((square[move_coordinates[0]][move_coordinates[1]] >= 0) && (square[move_coordinates[0]][move_coordinates[1]] != 6)) // if move square isn't black or white king
	{
		BlackPawnMovement(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, invalid_move, mate_checking);
	} // end of if move square isn't black or white king
} // end of ApplyBlackPawnMoves function

/* This function tries all knight moves */
void TryKnightMoves(int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state)
{
	/* Move 2 right, 1 up */
	if (piece_coordinates[0] + 2 <= 7 && piece_coordinates[1] + 1 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] + 2;
		move_coordinates[1] = piece_coordinates[1] + 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKnightMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 2 right, 1 down */
	if (piece_coordinates[0] + 2 <= 7 && piece_coordinates[1] - 1 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] + 2;
		move_coordinates[1] = piece_coordinates[1] - 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKnightMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 2 left, 1 up */
	if (piece_coordinates[0] - 2 >= 0 && piece_coordinates[1] + 1 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] - 2;
		move_coordinates[1] = piece_coordinates[1] + 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKnightMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 2 left, 1 down */
	if (piece_coordinates[0] - 2 >= 0 && piece_coordinates[1] - 1 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] - 2;
		move_coordinates[1] = piece_coordinates[1] - 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKnightMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 right, 2 up */
	if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] + 2 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] + 1;
		move_coordinates[1] = piece_coordinates[1] + 2;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKnightMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 right, 2 down */
	if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] - 2 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] + 1;
		move_coordinates[1] = piece_coordinates[1] - 2;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKnightMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 left, 2 up */
	if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] + 2 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] - 1;
		move_coordinates[1] = piece_coordinates[1] + 2;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKnightMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 left, 2 down */
	if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] - 2 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] - 1;
		move_coordinates[1] = piece_coordinates[1] - 2;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKnightMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

} // end of TryKnightMoves function

/* This function applies knight moves */
void ApplyKnightMoves(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move)
{
	if (square[piece_coordinates[0]][piece_coordinates[1]] == 2) // if piece is white knight
	{
		if ((square[move_coordinates[0]][move_coordinates[1]] <= 0) && (square[move_coordinates[0]][move_coordinates[1]] != -6)) // if move square isn't white or black king
		{
			KnightMovement(square, piece_coordinates, move_coordinates, invalid_move);
		} // end of if move square isn't white or black king
	} // end of if piece is white knight
	else if (square[piece_coordinates[0]][piece_coordinates[1]] == -2) // if piece is black knight
	{
		if ((square[move_coordinates[0]][move_coordinates[1]] >= 0) && (square[move_coordinates[0]][move_coordinates[1]] != 6)) // if move square isn't black or white king
		{
			KnightMovement(square, piece_coordinates, move_coordinates, invalid_move);
		} // end of if move square isn't black or white king
	} // end of if piece is black knight
} // end of ApplyKnightMoves function

/* This function tries all bishop moves */
void TryBishopMoves(int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state)
{
	int i, j;

	/* Move right and up */
	for (i = piece_coordinates[0] + 1; i < 8; i++)
	{
		j = piece_coordinates[1] + (i - piece_coordinates[0]);
		if (j <= 7)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			(*moves_tried)++;

			(*invalid_move) = 1;
			ApplyBishopMoves(square, piece_coordinates, move_coordinates, invalid_move);

			if ((*invalid_move) == 0) // if move is legal
			{
				CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
			} // end of if move is legal

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

	/* Move left and up */
	for (i = piece_coordinates[0] - 1; i >= 0; i--)
	{
		j = piece_coordinates[1] - (i - piece_coordinates[0]);
		if (j <= 7)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			(*moves_tried)++;

			(*invalid_move) = 1;
			ApplyBishopMoves(square, piece_coordinates, move_coordinates, invalid_move);

			if ((*invalid_move) == 0) // if move is legal
			{
				CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
			} // end of if move is legal

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

	/* Move left and down */
	for (i = piece_coordinates[0] - 1; i >= 0; i--)
	{
		j = piece_coordinates[1] + (i - piece_coordinates[0]);
		if (j >= 0)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			(*moves_tried)++;

			(*invalid_move) = 1;
			ApplyBishopMoves(square, piece_coordinates, move_coordinates, invalid_move);

			if ((*invalid_move) == 0) // if move is legal
			{
				CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
			} // end of if move is legal

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

	/* Move right and down */
	for (i = piece_coordinates[0] + 1; i < 8; i++)
	{
		j = piece_coordinates[1] - (i - piece_coordinates[0]);
		if (j >= 0)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			(*moves_tried)++;

			(*invalid_move) = 1;
			ApplyBishopMoves(square, piece_coordinates, move_coordinates, invalid_move);

			if ((*invalid_move) == 0) // if move is legal
			{
				CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
			} // end of if move is legal

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

} // end of TryBishopMoves function

/* This function applies bishop moves */
void ApplyBishopMoves(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move)
{
	if (square[piece_coordinates[0]][piece_coordinates[1]] == 3 || square[piece_coordinates[0]][piece_coordinates[1]] == 5) // if piece is white bishop or white queen
	{
		if ((square[move_coordinates[0]][move_coordinates[1]] <= 0) && (square[move_coordinates[0]][move_coordinates[1]] != -6)) // if move square isn't white or black king
		{
			BishopMovement(square, piece_coordinates, move_coordinates, invalid_move);
		} // end of if move square isn't white or black king
	} // end of if piece is white bishop or white queen
	else if (square[piece_coordinates[0]][piece_coordinates[1]] == -3 || square[piece_coordinates[0]][piece_coordinates[1]] == -5) // if piece is black bishop or black queen
	{
		if ((square[move_coordinates[0]][move_coordinates[1]] >= 0) && (square[move_coordinates[0]][move_coordinates[1]] != 6)) // if move square isn't black or white king
		{
			BishopMovement(square, piece_coordinates, move_coordinates, invalid_move);
		} // end of if move square isn't black or white king
	} // end of if piece is black bishop or black queen
} // end of ApplyBishopMoves function

/* This function tries all rook moves */
void TryRookMoves(int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state)
{
	int i;

	/* Move right */
	for (i = piece_coordinates[0] + 1; i < 8; i++)
	{
		move_coordinates[0] = i;
		move_coordinates[1] = piece_coordinates[1];

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyRookMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal

		if (square[i][piece_coordinates[1]] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop

	/* Move up */
	for (i = piece_coordinates[1] + 1; i < 8; i++)
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = i;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyRookMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal

		if (square[piece_coordinates[0]][i] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop

	/* Move left */
	for (i = piece_coordinates[0] - 1; i >= 0; i--)
	{
		move_coordinates[0] = i;
		move_coordinates[1] = piece_coordinates[1];

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyRookMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal

		if (square[i][piece_coordinates[1]] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop

	/* Move down */
	for (i = piece_coordinates[1] - 1; i >= 0; i--)
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = i;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyRookMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal

		if (square[piece_coordinates[0]][i] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop

} // end of TryRookMoves function

/* This function applies rook moves */
void ApplyRookMoves(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move)
{
	if (square[piece_coordinates[0]][piece_coordinates[1]] == 4 || square[piece_coordinates[0]][piece_coordinates[1]] == 5) // if piece is white rook or white queen
	{
		if ((square[move_coordinates[0]][move_coordinates[1]] <= 0) && (square[move_coordinates[0]][move_coordinates[1]] != -6)) // if move square isn't white or black king
		{
			RookMovement(square, piece_coordinates, move_coordinates, invalid_move);
		} // end of if move square isn't white or black king
	} // end of if piece is white rook or white queen
	else if (square[piece_coordinates[0]][piece_coordinates[1]] == -4 || square[piece_coordinates[0]][piece_coordinates[1]] == -5) // if piece is black rook or black queen
	{
		if ((square[move_coordinates[0]][move_coordinates[1]] >= 0) && (square[move_coordinates[0]][move_coordinates[1]] != 6)) // if move square isn't black or white king
		{
			RookMovement(square, piece_coordinates, move_coordinates, invalid_move);
		} // end of if move square isn't black or white king
	} // end of if piece is black rook or black queen
} // end of ApplyRookMoves function

/* This function tries all queen moves */
void TryQueenMoves(int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state)
{
	TryBishopMoves(piece_coordinates, move_coordinates, square, pawn_en_passant_status, invalid_move, moves_tried, moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, computer, turn, depth, piece_number, old_piece_square, old_move_square, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_board_state, old_board_state);

	TryRookMoves(piece_coordinates, move_coordinates, square, pawn_en_passant_status, invalid_move, moves_tried, moves_valid, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, computer, turn, depth, piece_number, old_piece_square, old_move_square, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_board_state, old_board_state);

} // end of TryQueenMoves function

/* This function tries all king moves */
void TryKingMoves(struct BoardState *board_state, int *piece_coordinates, int *move_coordinates, int **square, int **pawn_en_passant_status, int *invalid_move, unsigned int *moves_tried, unsigned int *moves_valid, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int computer, int turn, int depth, int piece_number, int *old_piece_square, int *old_move_square, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, struct BoardState *best_board_state, struct BoardState *old_board_state)
{
	/* Move 1 right */
	if (piece_coordinates[0] + 1 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] + 1;
		move_coordinates[1] = piece_coordinates[1];

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKingMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 right and up */
	if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] + 1 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] + 1;
		move_coordinates[1] = piece_coordinates[1] + 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKingMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 up */
	if (piece_coordinates[1] + 1 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = piece_coordinates[1] + 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKingMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 left and up */
	if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] + 1 <= 7) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] - 1;
		move_coordinates[1] = piece_coordinates[1] + 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKingMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 left */
	if (piece_coordinates[0] - 1 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] - 1;
		move_coordinates[1] = piece_coordinates[1];

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKingMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 left and down */
	if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] - 1 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] - 1;
		move_coordinates[1] = piece_coordinates[1] - 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKingMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 down */
	if (piece_coordinates[1] - 1 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = piece_coordinates[1] - 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKingMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Move 1 right and down */
	if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] - 1 >= 0) // if still on board
	{
		move_coordinates[0] = piece_coordinates[0] + 1;
		move_coordinates[1] = piece_coordinates[1] - 1;

		(*moves_tried)++;

		(*invalid_move) = 1;
		ApplyKingMoves(square, piece_coordinates, move_coordinates, invalid_move);

		if ((*invalid_move) == 0) // if move is legal
		{
			CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
		} // end of if move is legal
	} // end of if still on board

	/* Castling */
	if (square[piece_coordinates[0]][piece_coordinates[1]] == 6) // if square is white king
	{
		if (board_state->white_in_check == 0) // if white is NOT in check
		{
			if (piece_coordinates[0] == 4 && piece_coordinates[1] == 0) // if white king is at original position
			{
				move_coordinates[0] = piece_coordinates[0] - 2;
				move_coordinates[1] = piece_coordinates[1];

				(*moves_tried)++;

				(*invalid_move) = 1;
				if ((square[move_coordinates[0]][move_coordinates[1]] <= 0) && (square[move_coordinates[0]][move_coordinates[1]] != -6)) // if move square isn't white or black king
				{
					WhiteQueenSideCastling(board_state, piece_coordinates, move_coordinates, square, invalid_move);
				} // end of if move square isn't white or black king

				if ((*invalid_move) == 0) // if move is legal
				{
					CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
				} // end of if move is legal

				move_coordinates[0] = piece_coordinates[0] + 2;
				move_coordinates[1] = piece_coordinates[1];

				(*moves_tried)++;

				(*invalid_move) = 1;
				if ((square[move_coordinates[0]][move_coordinates[1]] <= 0) && (square[move_coordinates[0]][move_coordinates[1]] != -6)) // if move square isn't white or black king
				{
					WhiteKingSideCastling(board_state, piece_coordinates, move_coordinates, square, invalid_move);
				} // end of if move square isn't white or black king

				if ((*invalid_move) == 0) // if move is legal
				{
					CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
				} // end of if move is legal
			} // end of if white king is at original position
		} // end of if white is NOT in check
	} // end of if square is white king
	else if (square[piece_coordinates[0]][piece_coordinates[1]] == -6) // if square is black king
	{
		if (board_state->black_in_check == 0) // if black is NOT in check
		{
			if (piece_coordinates[0] == 4 && piece_coordinates[1] == 7) // if black king is at original position
			{
				move_coordinates[0] = piece_coordinates[0] - 2;
				move_coordinates[1] = piece_coordinates[1];

				(*moves_tried)++;

				(*invalid_move) = 1;
				if ((square[move_coordinates[0]][move_coordinates[1]] >= 0) && (square[move_coordinates[0]][move_coordinates[1]] != 6)) // if move square isn't black or white king
				{
					BlackQueenSideCastling(board_state, piece_coordinates, move_coordinates, square, invalid_move);
				} // end of if move square isn't black or white king

				if ((*invalid_move) == 0) // if move is legal
				{
					CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
				} // end of if move is legal

				move_coordinates[0] = piece_coordinates[0] + 2;
				move_coordinates[1] = piece_coordinates[1];

				(*moves_tried)++;

				(*invalid_move) = 1;
				if ((square[move_coordinates[0]][move_coordinates[1]] >= 0) && (square[move_coordinates[0]][move_coordinates[1]] != 6)) // if move square isn't black or white king
				{
					BlackKingSideCastling(board_state, piece_coordinates, move_coordinates, square, invalid_move);
				} // end of if move square isn't black or white king

				if ((*invalid_move) == 0) // if move is legal
				{
					CheckHumanMoveValidityForCheck(piece_coordinates, move_coordinates, square, piece_positions, invalid_move, moves_valid);
				} // end of if move is legal
			} // end of if black king is at original position
		} // end of if black is NOT in check
	} // end of if square is black king

} // end of TryKingMoves function

/* This function applies king moves */
void ApplyKingMoves(int **square, int *piece_coordinates, int *move_coordinates, int *invalid_move)
{
	if (square[piece_coordinates[0]][piece_coordinates[1]] == 6) // if piece is white king
	{
		if ((square[move_coordinates[0]][move_coordinates[1]] <= 0) && (square[move_coordinates[0]][move_coordinates[1]] != -6)) // if move square isn't white or black king
		{
			KingMovement(square, piece_coordinates, move_coordinates, invalid_move);
		} // end of if move square isn't white or black king
	} // end of if piece is white king
	else if (square[piece_coordinates[0]][piece_coordinates[1]] == -6) // if piece is white king
	{
		if ((square[move_coordinates[0]][move_coordinates[1]] >= 0) && (square[move_coordinates[0]][move_coordinates[1]] != 6)) // if move square isn't white or black king
		{
			KingMovement(square, piece_coordinates, move_coordinates, invalid_move);
		} // end of if move square isn't white or black king
	} // end of if piece is white king
} // end of ApplyKingMoves function

/******************************************************************************************************/
/********************************** SPECIAL GAME OVER CONDITIONS **************************************/
/******************************************************************************************************/

/* This function checks for three fold repetition */
int ThreeFoldRepetition(int turn, int **square, int ***board_history)
{
	unsigned int i, j, k, repetition, board_history_counter;
	int break_it = 0;

	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		for (j = 0; j < chess_constants.number_of_rows; j++)
		{
			board_history[i][j][turn] = square[i][j];
		} // end of j loop
	} // end of i loop

	repetition = 0;
	for (k = 0; k <= turn; k++)
	{
		board_history_counter = 0;
		for (i = 0; i < chess_constants.number_of_columns; i++)
		{
			for (j = 0; j < chess_constants.number_of_rows; j++)
			{
				if (square[i][j] == board_history[i][j][k]) // if square matches square's board_history
				{
					board_history_counter++;
				} // end of if square matches square's board_history
			} // end of j loop
		} // end of i loop

		if (board_history_counter == 64) // if all squares matched board_history
		{
			repetition++;
		} // end of if all squares matched board_history
	} // end of k loop

	if (repetition >= 3) // if a board state has happened at least 3 times
	{
		break_it = 1;
	} // end of if a board state has happened at least 3 times

	return break_it;
} // end of ThreeFoldRepetition function

/* This function checks the fifty move rule of no pawn move AND piece capture in fifty moves */
int FiftyMoveRule(struct BoardState *board_state, int *move_coordinates, int **square)
{
	int break_it = 0;

	if (square[move_coordinates[0]][move_coordinates[1]] == 1 || square[move_coordinates[0]][move_coordinates[1]] == -1) // if pawn moves then update pawn_move
	{
		board_state->pawn_move = board_state->turn;
	} // end of if pawn moves then update pawn_move

	if (board_state->white_pieces_old - board_state->white_pieces != 0 || board_state->black_pieces_old - board_state->black_pieces != 0) // if the number of pieces has changed
	{
		board_state->capture_move = board_state->turn;
	} // end of if the number of pieces has changed

	board_state->white_pieces_old = board_state->white_pieces;
	board_state->black_pieces_old = board_state->black_pieces;

	if (board_state->turn - board_state->pawn_move >= 50 * 2 && board_state->turn - board_state->capture_move >= 50 * 2) // if a pawn hasn't moved AND a capture hasn't happened in at least 50 turns
	{
		break_it = 1;
	} // end of if a pawn hasn't moved or a capture hasn't happened in at least 50 turns

	return break_it;
} // end of FiftyMoveRule function

/* This function checks if checmate is impossible */
int CheckmateImpossibility(unsigned int **piece_count, int **square, int **color)
{
	unsigned int i, j;
	int break_it = 0, bishop_color;

	if (piece_count[0][0] == 0 && piece_count[0][3] == 0 && piece_count[0][4] == 0) // if no more white pawns, rooks, and queens
	{
		if (piece_count[1][0] == 0 && piece_count[1][3] == 0 && piece_count[1][4] == 0) // if no more black pawns, rooks, and queens
		{
			if (piece_count[0][1] == 0 && piece_count[1][1] == 0) // if no more knights
			{
				if (piece_count[0][2] == 0 && piece_count[1][2] == 0) // if no more bishops
				{
					break_it = 1;  // Checkmate is impossible! It is a draw
				} // end of if no more bishops
				else if (piece_count[0][2] == 1 || piece_count[1][2] == 0) // if only white bishop plus kings
				{
					break_it = 1; // Checkmate is impossible! It is a draw
				} // end of if only white bishop plus kings
				else if (piece_count[0][2] == 0 || piece_count[1][2] == 1) // if only black bishop plus kings
				{
					break_it = 1; // Checkmate is impossible! It is a draw
				} // end of if only black bishop plus kings
				else if (piece_count[0][2] == 1 && piece_count[1][2] == 1) // if there is a bishop and king each
				{
					for (i = 0; i < chess_constants.number_of_columns; i++)
					{
						for (j = 0; j < chess_constants.number_of_rows; j++)
						{
							if (square[i][j] == 3) // if square is white bishop
							{
								bishop_color = color[i][j];
							} // end of if square is white bishop
							else if (square[i][j] == -3) // if square is black bishop
							{
								if (color[i][j] == bishop_color) // if bishops are on same color square
								{
									break_it = 1; // Checkmate is impossible! It is a draw
									break;
								} // end of if bishops are on same color square
							} // end of if square is black bishop
						} // end of j loop

						if (break_it == 1)
						{
							break;
						}
					} // end of i loop
				} // end of if there is a bishop and king each
			} // end of if no more knights
			else if ((piece_count[0][1] == 1) && (piece_count[1][1] == 0)) // if just one knight
			{
				if ((piece_count[0][2] == 0) && (piece_count[1][2] == 0)) // if no more bishops so just a knight and kings
				{
					break_it = 1; // Checkmate is impossible! It is a draw
				} // end of if no more bishops so just a knight and kings
			} // end of if just one knight
			else if ((piece_count[0][1] == 0) && (piece_count[1][1] == 1)) // if just one knight
			{
				if ((piece_count[0][2] == 0) && (piece_count[1][2] == 0)) // if no more bishops so just a knight and kings
				{
					break_it = 1; // Checkmate is impossible! It is a draw
				} // end of if no more bishops so just a knight and kings
			} // end of if just one knight
		} // end of if no more black pawns, rooks, and queens
	} // end of if no more white pawns, rooks, and queens

	return break_it;
} // end of CheckmateImpossibility function

/******************************************************************************************************/
/************************************* BOARD EVALUATION SCORE *****************************************/
/******************************************************************************************************/

/* This function evaluates the boards score */
int BoardEvaluationScore(struct BoardState *board_state, int turn, int **square, int **color, int ****piece_positions, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers)
{
	unsigned int i, j;
	int pawn_not_passed = 0, pawn_connected = 0, white_pawns_in_middle = 0, black_pawns_in_middle = 0, white_pawns_not_passed = 0, black_pawns_not_passed = 0;
	int white_pawns_on_white_squares = 0, white_pawns_on_black_squares = 0, black_pawns_on_white_squares = 0, black_pawns_on_black_squares = 0;
	int evaluation_score = 0, game_over = 0;

	/* Add up pawn values */
	if (board_state->white_pieces + board_state->black_pieces >= 8) // if not end_game yet
	{
		board_state->end_game = 0;

		/* White pawn values */
		for (i = 0; i < piece_count[0][0]; i++)
		{
			if (piece_positions[0][0][i][1] >= 3 && piece_positions[0][0][i][1] <= 6) // if pawn is in the 4th through 7th ranks
			{
				pawn_not_passed = 0;
				for (j = 0; j < piece_count[1][0]; j++)
				{
					if (abs(piece_positions[1][0][j][0] - piece_positions[0][0][i][0]) == 0 || abs(piece_positions[1][0][j][0] - piece_positions[0][0][i][0]) == 1) // if there is an opposing pawn in same file or an adjacent one
					{
						if (piece_positions[1][0][j][1] > piece_positions[0][0][i][1]) // if opposing pawn is still ahead of current pawn
						{
							pawn_not_passed = 1;
							break;
						} // end of if opposing pawn is still ahead of current pawn
					} // end of if there is an opposing pawn in same file or an adjacent one
				} // end of j loop

				if (pawn_not_passed == 1) // if pawn is NOT passed
				{
					pawn_connected = 0;
					for (j = 0; j < piece_count[0][0]; j++)
					{
						if (abs(piece_positions[0][0][j][0] - piece_positions[0][0][i][0]) == 1) // if there is a friendly pawn in an adjacent file
						{
							if (abs(piece_positions[0][0][j][1] - piece_positions[0][0][i][1]) == 0 || abs(piece_positions[0][0][j][1] - piece_positions[0][0][i][1]) == 1) // if jth pawn is in the same rank as ith pawn or an adjacent one
							{
								pawn_connected = 1;
								break;
							} // end of if jth pawn is in the same rank as ith pawn or an adjacent one
						} // end of if there is a friendly pawn in an adjacent file
					} // end of j loop

					if (pawn_connected == 1) // if pawn is connected
					{
						evaluation_score = evaluation_score + (int)(non_passed_pawn_opening_values[piece_positions[0][0][i][0]][piece_positions[0][0][i][1]] * pawn_advancement_multipliers[1][piece_positions[0][0][i][1] - 3]);
					} // end of if pawn is connected
					else // if pawn is NOT connected
					{
						evaluation_score = evaluation_score + (int)(non_passed_pawn_opening_values[piece_positions[0][0][i][0]][piece_positions[0][0][i][1]] * pawn_advancement_multipliers[0][piece_positions[0][0][i][1] - 3]);
					} // if pawn is NOT connected
				} // end of if pawn is NOT passed
				else // if pawn is passed
				{
					pawn_connected = 0;
					for (j = 0; j < piece_count[0][0]; j++)
					{
						if (abs(piece_positions[0][0][j][0] - piece_positions[0][0][i][0]) == 1) // if there is a friendly pawn in an adjacent file
						{
							if (abs(piece_positions[0][0][j][1] - piece_positions[0][0][i][1]) == 0 || abs(piece_positions[0][0][j][1] - piece_positions[0][0][i][1]) == 1) // if jth pawn is in the same rank as ith pawn or an adjacent one
							{
								pawn_connected = 1;
								break;
							} // end of if jth pawn is in the same rank as ith pawn or an adjacent one
						} // end of if there is a friendly pawn in an adjacent file
					} // end of j loop

					if (pawn_connected == 1) // if pawn is connected
					{
						evaluation_score = evaluation_score + (int)(piece_value[0] * pawn_advancement_multipliers[3][piece_positions[0][0][i][1] - 3]);
					} // end of if pawn is connected
					else // if pawn is NOT connected
					{
						evaluation_score = evaluation_score + (int)(piece_value[0] * pawn_advancement_multipliers[2][piece_positions[0][0][i][1] - 3]);
					} // if pawn is NOT connected
				} // end of if pawn is passed
			} // end of if pawn is in the 4th through 7th ranks
			else // if pawn is NOT in the 4th through 7th ranks
			{
				evaluation_score = evaluation_score+non_passed_pawn_opening_values[piece_positions[0][0][i][0]][piece_positions[0][0][i][1]];
			} // end of if pawn is NOT in the 4th through 7th ranks
		} // end of i loop

		/* Black pawn values */
		for (i = 0; i < piece_count[1][0]; i++)
		{
			if (piece_positions[1][0][i][1] >= 1 && piece_positions[1][0][i][1] <= 4) // if pawn is in the 4th through 7th ranks
			{
				pawn_not_passed = 0;
				for (j = 0; j < piece_count[0][0]; j++)
				{
					if (abs(piece_positions[0][0][j][0] - piece_positions[1][0][i][0]) == 0 || abs(piece_positions[0][0][j][0] - piece_positions[1][0][i][0]) == 1) // if there is an opposing pawn in same file or an adjacent one
					{
						if (piece_positions[0][0][j][1] < piece_positions[1][0][i][1]) // if opposing pawn is still ahead of current pawn
						{
							pawn_not_passed = 1;
							break;
						} // end of if opposing pawn is still ahead of current pawn
					} // end of if there is an opposing pawn in same file or an adjacent one
				} // end of j loop

				if (pawn_not_passed == 1) // if pawn is NOT passed
				{
					pawn_connected = 0;
					for (j = 0; j < piece_count[1][0]; j++)
					{
						if (abs(piece_positions[1][0][j][0] - piece_positions[1][0][i][0]) == 1) // if there is a friendly pawn in an adjacent file
						{
							if (abs(piece_positions[1][0][j][1] - piece_positions[1][0][i][1]) == 0 || abs(piece_positions[1][0][j][1] - piece_positions[1][0][i][1]) == 1) // if jth pawn is in the same rank as ith pawn or an adjacent one
							{
								pawn_connected = 1;
								break;
							} // end of if jth pawn is in the same rank as ith pawn or an adjacent one
						} // end of if there is a friendly pawn in an adjacent file
					} // end of j loop

					if (pawn_connected == 1) // if pawn is connected
					{
						evaluation_score = evaluation_score - (int)(non_passed_pawn_opening_values[piece_positions[1][0][i][0]][7-piece_positions[1][0][i][1]] * pawn_advancement_multipliers[1][7-piece_positions[1][0][i][1] - 3]);
					} // end of if pawn is connected
					else // if pawn is NOT connected
					{
						evaluation_score = evaluation_score - (int)(non_passed_pawn_opening_values[piece_positions[1][0][i][0]][7-piece_positions[1][0][i][1]] * pawn_advancement_multipliers[0][7-piece_positions[1][0][i][1] - 3]);
					} // if pawn is NOT connected
				} // end of if pawn is NOT passed
				else // if pawn is passed
				{
					pawn_connected = 0;
					for (j = 0; j < piece_count[1][0]; j++)
					{
						if (abs(piece_positions[1][0][j][0] - piece_positions[1][0][i][0]) == 1) // if there is a friendly pawn in an adjacent file
						{
							if (abs(piece_positions[1][0][j][1] - piece_positions[1][0][i][1]) == 0 || abs(piece_positions[1][0][j][1] - piece_positions[1][0][i][1]) == 1) // if jth pawn is in the same rank as ith pawn or an adjacent one
							{
								pawn_connected = 1;
								break;
							} // end of if jth pawn is in the same rank as ith pawn or an adjacent one
						} // end of if there is a friendly pawn in an adjacent file
					} // end of j loop

					if (pawn_connected == 1) // if pawn is connected
					{
						evaluation_score = evaluation_score - (int)(piece_value[0] * pawn_advancement_multipliers[3][7-piece_positions[1][0][i][1] - 3]);
					} // end of if pawn is connected
					else // if pawn is NOT connected
					{
						evaluation_score = evaluation_score - (int)(piece_value[0] * pawn_advancement_multipliers[2][7-piece_positions[1][0][i][1] - 3]);
					} // if pawn is NOT connected
				} // end of if pawn is passed
			} // end of if pawn is in the 4th through 7th ranks
			else // if pawn is NOT in the 4th through 7th ranks
			{
				evaluation_score = evaluation_score - non_passed_pawn_opening_values[piece_positions[1][0][i][0]][7-piece_positions[1][0][i][1]];
			} // end of if pawn is NOT in the 4th through 7th ranks
		} // end of i loop
	} // end of if not end_game yet
	else // if end_game
	{
		board_state->end_game = 1;

		/* White pawn values */
		for (i = 0; i < piece_count[0][0]; i++)
		{
			if (piece_positions[0][0][i][1] >= 3 && piece_positions[0][0][i][1] <= 6) // if pawn is in the 4th through 7th ranks
			{
				pawn_not_passed = 0;
				for (j = 0; j < piece_count[1][0]; j++)
				{
					if (abs(piece_positions[1][0][j][0] - piece_positions[0][0][i][0]) == 0 || abs(piece_positions[1][0][j][0] - piece_positions[0][0][i][0]) == 1) // if there is an opposing pawn in same file or an adjacent one
					{
						if (piece_positions[1][0][j][1] > piece_positions[0][0][i][1]) // if opposing pawn is still ahead of current pawn
						{
							pawn_not_passed = 1;
							break;
						} // end of if opposing pawn is still ahead of current pawn
					} // end of if there is an opposing pawn in same file or an adjacent one
				} // end of j loop

				if (pawn_not_passed == 1) // if pawn is NOT passed
				{
					pawn_connected = 0;
					for (j = 0; j < piece_count[0][0]; j++)
					{
						if (abs(piece_positions[0][0][j][0] - piece_positions[0][0][i][0]) == 1) // if there is a friendly pawn in an adjacent file
						{
							if (abs(piece_positions[0][0][j][1] - piece_positions[0][0][i][1]) == 0 || abs(piece_positions[0][0][j][1] - piece_positions[0][0][i][1]) == 1) // if jth pawn is in the same rank as ith pawn or an adjacent one
							{
								pawn_connected = 1;
								break;
							} // end of if jth pawn is in the same rank as ith pawn or an adjacent one
						} // end of if there is a friendly pawn in an adjacent file
					} // end of j loop

					if (pawn_connected == 1) // if pawn is connected
					{
						evaluation_score = evaluation_score + (int)(non_passed_pawn_end_game_values[piece_positions[0][0][i][0]][piece_positions[0][0][i][1]] * pawn_advancement_multipliers[1][piece_positions[0][0][i][1] - 3]);
					} // end of if pawn is connected
					else // if pawn is NOT connected
					{
						evaluation_score = evaluation_score + (int)(non_passed_pawn_end_game_values[piece_positions[0][0][i][0]][piece_positions[0][0][i][1]] * pawn_advancement_multipliers[0][piece_positions[0][0][i][1] - 3]);
					} // if pawn is NOT connected
				} // end of if pawn is NOT passed
				else // if pawn is passed
				{
					pawn_connected = 0;
					for (j = 0; j < piece_count[0][0]; j++)
					{
						if (abs(piece_positions[0][0][j][0] - piece_positions[0][0][i][0]) == 1) // if there is a friendly pawn in an adjacent file
						{
							if (abs(piece_positions[0][0][j][1] - piece_positions[0][0][i][1]) == 0 || abs(piece_positions[0][0][j][1] - piece_positions[0][0][i][1]) == 1) // if jth pawn is in the same rank as ith pawn or an adjacent one
							{
								pawn_connected = 1;
								break;
							} // end of if jth pawn is in the same rank as ith pawn or an adjacent one
						} // end of if there is a friendly pawn in an adjacent file
					} // end of j loop

					if (pawn_connected == 1) // if pawn is connected
					{
						evaluation_score = evaluation_score + (int)(piece_value[0] * pawn_advancement_multipliers[3][piece_positions[0][0][i][1] - 3]);
					} // end of if pawn is connected
					else // if pawn is NOT connected
					{
						evaluation_score = evaluation_score + (int)(piece_value[0] * pawn_advancement_multipliers[2][piece_positions[0][0][i][1] - 3]);
					} // if pawn is NOT connected
				} // end of if pawn is passed
			} // end of if pawn is in the 4th through 7th ranks
			else // if pawn is NOT in the 4th through 7th ranks
			{
				evaluation_score = evaluation_score+non_passed_pawn_end_game_values[piece_positions[0][0][i][0]][piece_positions[0][0][i][1]];
			} // end of if pawn is NOT in the 4th through 7th ranks
		} // end of i loop

		/* Black pawn values */
		for (i = 0; i < piece_count[1][0]; i++)
		{
			if (piece_positions[1][0][i][1] >= 1 && piece_positions[1][0][i][1] <= 4) // if pawn is in the 4th through 7th ranks
			{
				pawn_not_passed = 0;
				for (j = 0; j < piece_count[0][0]; j++)
				{
					if (abs(piece_positions[0][0][j][0] - piece_positions[1][0][i][0]) == 0 || abs(piece_positions[0][0][j][0] - piece_positions[1][0][i][0]) == 1) // if there is an opposing pawn in same file or an adjacent one
					{
						if (piece_positions[0][0][j][1] < piece_positions[1][0][i][1]) // if opposing pawn is still ahead of current pawn
						{
							pawn_not_passed = 1;
							break;
						} // end of if opposing pawn is still ahead of current pawn
					} // end of if there is an opposing pawn in same file or an adjacent one
				} // end of j loop

				if (pawn_not_passed == 1) // if pawn is NOT passed
				{
					pawn_connected = 0;
					for (j = 0; j < piece_count[1][0]; j++)
					{
						if (abs(piece_positions[1][0][j][0] - piece_positions[1][0][i][0]) == 1) // if there is a friendly pawn in an adjacent file
						{
							if (abs(piece_positions[1][0][j][1] - piece_positions[1][0][i][1]) == 0 || abs(piece_positions[1][0][j][1] - piece_positions[1][0][i][1]) == 1) // if jth pawn is in the same rank as ith pawn or an adjacent one
							{
								pawn_connected = 1;
								break;
							} // end of if jth pawn is in the same rank as ith pawn or an adjacent one
						} // end of if there is a friendly pawn in an adjacent file
					} // end of j loop

					if (pawn_connected == 1) // if pawn is connected
					{
						evaluation_score = evaluation_score - (int)(non_passed_pawn_end_game_values[piece_positions[1][0][i][0]][7-piece_positions[1][0][i][1]] * pawn_advancement_multipliers[1][7-piece_positions[1][0][i][1] - 3]);
					} // end of if pawn is connected
					else // if pawn is NOT connected
					{
						evaluation_score = evaluation_score - (int)(non_passed_pawn_end_game_values[piece_positions[1][0][i][0]][7-piece_positions[1][0][i][1]] * pawn_advancement_multipliers[0][7-piece_positions[1][0][i][1] - 3]);
					} // if pawn is NOT connected
				} // end of if pawn is NOT passed
				else // if pawn is passed
				{
					pawn_connected = 0;
					for (j = 0; j < piece_count[1][0]; j++)
					{
						if (abs(piece_positions[1][0][j][0] - piece_positions[1][0][i][0]) == 1) // if there is a friendly pawn in an adjacent file
						{
							if (abs(piece_positions[1][0][j][1] - piece_positions[1][0][i][1]) == 0 || abs(piece_positions[1][0][j][1] - piece_positions[1][0][i][1]) == 1) // if jth pawn is in the same rank as ith pawn or an adjacent one
							{
								pawn_connected = 1;
								break;
							} // end of if jth pawn is in the same rank as ith pawn or an adjacent one
						} // end of if there is a friendly pawn in an adjacent file
					} // end of j loop

					if (pawn_connected == 1) // if pawn is connected
					{
						evaluation_score = evaluation_score - (int)(piece_value[0] * pawn_advancement_multipliers[3][7-piece_positions[1][0][i][1] - 3]);
					} // end of if pawn is connected
					else // if pawn is NOT connected
					{
						evaluation_score = evaluation_score - (int)(piece_value[0] * pawn_advancement_multipliers[2][7-piece_positions[1][0][i][1] - 3]);
					} // if pawn is NOT connected
				} // end of if pawn is passed
			} // end of if pawn is in the 4th through 7th ranks
			else // if pawn is NOT in the 4th through 7th ranks
			{
				evaluation_score = evaluation_score - non_passed_pawn_end_game_values[piece_positions[1][0][i][0]][7-piece_positions[1][0][i][1]];
			} // end of if pawn is NOT in the 4th through 7th ranks
		} // end of i loop
	} // end of if end_game

	if (white_pawns_in_middle >= 4 && black_pawns_in_middle >= 4 && white_pawns_not_passed >= 3 && black_pawns_not_passed >= 3) // if there is pawn congestion in the middle
	{
		board_state->closed_position = 1;
	} // end of if there is pawn congestion in the middle
	else // if there is NOT pawn congestion in the middle
	{
		board_state->closed_position = 0;
	} // end of if there is NOT pawn congestion in the middle

	if (board_state->closed_position == 1) // if there is a closed position
	{
		/* White knights */
		for (i = 0; i < piece_count[0][1]; i++)
		{
			evaluation_score = evaluation_score + (int)(piece_value[1] * (1.5 + knight_position_modifiers[piece_positions[0][1][i][0]][piece_positions[0][1][i][1]]));
		} // end of i loop

		/* Black nights */
		for (i = 0; i < piece_count[1][1]; i++)
		{
			evaluation_score = evaluation_score - (int)(piece_value[1] * (1.5 + knight_position_modifiers[piece_positions[1][1][i][0]][piece_positions[1][1][i][1]]));
		} // end of i loop

		/* White bishops */
		for (i = 0; i < piece_count[0][2]; i++)
		{
			if (color[piece_positions[0][2][i][0]][piece_positions[0][2][i][1]] == 0) // if square's color is white
			{
				evaluation_score = evaluation_score + (int)(piece_value[2] * (0.8 - (white_pawns_on_white_squares - 4) * 0.0125));
			} // end of if square's color is white
			else // if square color is black
			{
				evaluation_score = evaluation_score + (int)(piece_value[2] * (0.8 - (white_pawns_on_black_squares - 4) * 0.0125));
			} // end of if square color is black
		} // end of i loop

		/* Black bishops */
		for (i = 0; i < piece_count[1][2]; i++)
		{
			if (color[piece_positions[1][2][i][0]][piece_positions[1][2][i][1]] == 0) // if square's color is white
			{
				// White bishops
				evaluation_score = evaluation_score - (int)(piece_value[2] * (0.8 - (black_pawns_on_white_squares - 4) * 0.0125));
			} // end of if square's color is white
			else // if square color is black
			{
				// White bishops
				evaluation_score = evaluation_score - (int)(piece_value[2] * (0.8 - (black_pawns_on_black_squares - 4) * 0.0125));
			} // end of if square color is black
		} // end of i loop

		/* White rooks */
		evaluation_score = evaluation_score + piece_count[0][3] * piece_value[3] * 0.8;

		/* Black rooks */
		evaluation_score = evaluation_score - piece_count[1][3] * piece_value[3] * 0.8;

		/* White queens */
		evaluation_score = evaluation_score + piece_count[0][4] * piece_value[4] * 0.8;

		/* Black queens */
		evaluation_score = evaluation_score - piece_count[1][4] * piece_value[4] * 0.8;
	} // end of if there is a closed position
	else // if there is an open position
	{
		/* White knights */
		for (i = 0; i < piece_count[0][1]; i++)
		{
			evaluation_score = evaluation_score + (int)(piece_value[1] * (1.0 + knight_position_modifiers[piece_positions[0][1][i][0]][piece_positions[0][1][i][1]]));
		} // end of i loop

		/* Black nights */
		for (i = 0; i < piece_count[1][1]; i++)
		{
			evaluation_score = evaluation_score - (int)(piece_value[1] * (1.0 + knight_position_modifiers[piece_positions[1][1][i][0]][piece_positions[1][1][i][1]]));
		} // end of i loop

		/* White bishops */
		for (i = 0; i < piece_count[0][2]; i++)
		{
			if (color[piece_positions[0][2][i][0]][piece_positions[0][2][i][1]] == 0) // if square's color is white
			{
				evaluation_score = evaluation_score + (int)(piece_value[2] * (1.1 - (white_pawns_on_white_squares - 4) * 0.0125));
			} // end of if square's color is white
			else // if square color is black
			{
				evaluation_score = evaluation_score + (int)(piece_value[2] * (1.1 - (white_pawns_on_black_squares - 4) * 0.0125));
			} // end of if square color is black
		} // end of i loop

		/* Black bishops */
		for (i = 0; i < piece_count[1][2]; i++)
		{
			if (color[piece_positions[1][2][i][0]][piece_positions[1][2][i][1]] == 0) // if square's color is white
			{
				// White bishops
				evaluation_score = evaluation_score - (int)(piece_value[2] * (1.1 - (black_pawns_on_white_squares - 4) * 0.0125));
			} // end of if square's color is white
			else // if square color is black
			{
				// White bishops
				evaluation_score = evaluation_score - (int)(piece_value[2] * (1.1 - (black_pawns_on_black_squares - 4) * 0.0125));
			} // end of if square color is black
		} // end of i loop

		/* White rooks */
		evaluation_score = evaluation_score + piece_count[0][3] * piece_value[3] * 1.1;

		/* Black rooks */
		evaluation_score = evaluation_score - piece_count[1][3] * piece_value[3] * 1.1;

		/* White queens */
		evaluation_score = evaluation_score + piece_count[0][4] * piece_value[4] * 1.1;

		/* Black queens */
		evaluation_score = evaluation_score - piece_count[1][4] * piece_value[4] * 1.1;
	} // end of if there is an open position

	if (board_state->black_in_check == 1) // if black is in check
	{
		evaluation_score = evaluation_score + board_state->black_in_check * 1000;
	} // end of if white is in check

	if (board_state->white_in_check == 1) // if white is in check
	{
		evaluation_score = evaluation_score - board_state->white_in_check * 1000;
	} // end of if white is in check

	return evaluation_score;
} // end of BoardEvaluationScore function

/******************************************************************************************************/
/****************************************** COMPUTER TURN *********************************************/
/******************************************************************************************************/

/* This function is a computer player playing white */
int WhiteComputer(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct PlayerOptions *player_options, struct BoardState *board_state, int max_depth, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int **square, int **color, int ****piece_positions, unsigned int **piece_count, int *piece_value, int **pawn_en_passant_status, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int *piece_coordinates, int *move_coordinates, int ***board_history, char *basic_chessboard)
{
	ComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, board_state->turn, 0, max_depth, square, color, piece_positions, piece_count, piece_value, pawn_en_passant_status, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, piece_coordinates, move_coordinates, board_history);

	CountPieces(board_state, square, piece_count, piece_positions);

	board_state->white_in_check = 0;
	board_state->black_in_check = InCheckChecker(square, piece_positions[1][5][0], -1);

	if (board_state->black_in_check == 1) // if black is now in check
	{
		printf("Black is now in check!\n");
	} // end of if black is now in check

	PrintPieces(board_state->turn, square, color, player_options->players, basic_chessboard);

	if (CheckIfGameOver(board_state, board_state->turn, square, pawn_en_passant_status, piece_count, move_coordinates, piece_positions, color, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, 0) == 1)
	{
		return board_state->game_over;
	}

	/******************************************************************************************************/
	/********************************* CALCULATE BOARD EVALUATION SCORE ***********************************/
	/******************************************************************************************************/

	board_state->evaluation_score = BoardEvaluationScore(board_state, board_state->turn, square, color, piece_positions, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers);

	PrintPositionGamePhaseScore(board_state);

	return board_state->game_over;
} // end of WhiteComputer function

/* This function is a computer player playing white */
int BlackComputer(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct PlayerOptions *player_options, struct BoardState *board_state, int max_depth, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int **square, int **color, int ****piece_positions, unsigned int **piece_count, int *piece_value, int **pawn_en_passant_status, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int *piece_coordinates, int *move_coordinates, int ***board_history, char *basic_chessboard)
{
	ComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, board_state->turn, 0, max_depth, square, color, piece_positions, piece_count, piece_value, pawn_en_passant_status, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, piece_coordinates, move_coordinates, board_history);

	CountPieces(board_state, square, piece_count, piece_positions);

	board_state->black_in_check = 0;
	board_state->white_in_check = InCheckChecker(square, piece_positions[0][5][0], 1);

	if (board_state->white_in_check == 1) // if white is now in check
	{
		printf("White is now in check!\n");
	} // end of if white is now in check

	PrintPieces(board_state->turn, square, color, player_options->players, basic_chessboard);

	if (CheckIfGameOver(board_state, board_state->turn, square, pawn_en_passant_status, piece_count, move_coordinates, piece_positions, color, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, 0) == 1)
	{
		return board_state->game_over;
	}

	//---------------------------------------------------------------Calculate board evalulation score---------------------------------------------------------------

	board_state->evaluation_score = BoardEvaluationScore(board_state, board_state->turn, square, color, piece_positions, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers);

	PrintPositionGamePhaseScore(board_state);

	return board_state->game_over;
} // end of BlackComputer function

/* This function performs computer moves */
int ComputerMoves(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int **color, int ****piece_positions, unsigned int **piece_count, int *piece_value, int **pawn_en_passant_status, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int *piece_coordinates, int *move_coordinates, int ***board_history)
{
	int i, j, piece_file = -9, piece_rank = -9, move_found = 0;
	int old_piece_square = -9, old_move_square = -9;
	int best_score = -999999999, best_piece_file = -9, best_piece_rank = -9, best_move_file = -9, best_move_rank = -9, best_promotion = -9;

	/* Create a structure to store the current board state before we change the working one */
	struct BoardState old_board_state;
	old_board_state = (*board_state);

	/* Create a structure to store the best board state found from searching the game tree */
	struct BoardState best_board_state;
	InitializeBoardStateVariables(&best_board_state);

	int *alpha = malloc(sizeof(int) * max_depth);
	int *beta = malloc(sizeof(int) * max_depth);

	for (i = 0; i < max_depth; i++)
	{
		alpha[i] = -999999999;
		beta[i] = 999999999;
	} // end of i loop

	for (i = 0; i < chess_constants.number_of_unique_pieces; i++)
	{
		for (j = 0; j < piece_count[1 - (turn % 2)][i]; j++)
		{
			piece_file = piece_positions[1 - (turn % 2)][i][j][0];
			piece_rank = piece_positions[1 - (turn % 2)][i][j][1];

			piece_coordinates[0] = piece_file;
			piece_coordinates[1] = piece_rank;

			old_piece_square = square[piece_file][piece_rank];

			if (optimized_computer_moves == 1) // if using optimized computer moves
			{
				if (old_piece_square == 1) // if piece is a white pawn
				{
					MinimaxWhitePawn(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
				} // end of if piece is a white pawn
				else if (old_piece_square == -1) // if piece is a black pawn
				{
					MinimaxBlackPawn(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
				} // end of if piece is a pawn
				else if (abs(old_piece_square) == 2) // if piece is a knight
				{
					MinimaxKnight(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
				} // end of if piece is a knight
				else if (abs(old_piece_square) == 3) // if piece is a bishop
				{
					MinimaxBishop(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
				} // end of if piece is a bishop
				else if (abs(old_piece_square) == 4) // if piece is a rook
				{
					MinimaxRook(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
				} // end of if piece is a rook
				else if (abs(old_piece_square) == 5) // if piece is a queen
				{
					MinimaxQueen(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
				} // end of if piece is a queen
				else if (abs(old_piece_square) == 6) // if piece is a king
				{
					MinimaxKing(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
				} // end of if piece is a king
			} // end of if using optimized computer moves
			else // if NOT using optimized computer moves
			{
				MinimaxUnoptimized(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
			} // end of if NOT using optimized computer moves
		} // end of j loop
	} // end of i loop

	free(beta);
	free(alpha);

	if (best_move_file == -9)
	{
		printf("FAILED! best_score = %d, best_piece_file = %d, best_piece_rank = %d, best_move_file = %d, best_move_rank = %d, best_promotion = %d", best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion);
		printf(", best_white_queen_side_castle = %d, best_black_queen_side_castle = %d, best_white_king_side_castle = %d, best_black_king_side_castle = %d, best_white_queen_side_castle_allow = %d, best_black_queen_side_castle_allow = %d, best_white_king_side_castle_allow = %d, best_black_king_side_castle_allow = %d, best_en_passant_captured = %d, best_pawn_move = %u, best_capture_move = %u\n\n", best_board_state.white_queen_side_castle, best_board_state.black_queen_side_castle, best_board_state.white_king_side_castle, best_board_state.black_king_side_castle, best_board_state.white_queen_side_castle_allow, best_board_state.black_queen_side_castle_allow, best_board_state.white_king_side_castle_allow, best_board_state.black_king_side_castle_allow, best_board_state.en_passant_captured, best_board_state.pawn_move, best_board_state.capture_move);

		return 0;
	}
	else
	{
		piece_coordinates[0] = best_piece_file;
		piece_coordinates[1] = best_piece_rank;

		move_coordinates[0] = best_move_file;
		move_coordinates[1] = best_move_rank;

		printf("\nFINAL Piece at [%c%d] with square = %d moving to [%c%d] with square = %d\n", column_names[best_piece_file], best_piece_rank + 1, square[best_piece_file][best_piece_rank], column_names[best_move_file], best_move_rank + 1, square[best_move_file][best_move_rank]);
		printf("best_score = %d, best_piece_file = %d, best_piece_rank = %d, best_move_file = %d, best_move_rank = %d, best_promotion = %d", best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion);
		printf(", best_white_queen_side_castle = %d, best_black_queen_side_castle = %d, best_white_king_side_castle = %d, best_black_king_side_castle = %d, best_white_queen_side_castle_allow = %d, best_black_queen_side_castle_allow = %d, best_white_king_side_castle_allow = %d, best_black_king_side_castle_allow = %d, best_en_passant_captured = %d, best_pawn_move = %u, best_capture_move = %u\n\n", best_board_state.white_queen_side_castle, best_board_state.black_queen_side_castle, best_board_state.white_king_side_castle, best_board_state.black_king_side_castle, best_board_state.white_queen_side_castle_allow, best_board_state.black_queen_side_castle_allow, best_board_state.white_king_side_castle_allow, best_board_state.black_king_side_castle_allow, best_board_state.en_passant_captured, best_board_state.pawn_move, best_board_state.capture_move);

		if (square[best_piece_file][best_piece_rank] == 1) // if piece we moved was a white pawn
		{
			WhiteEnPassant(square, piece_coordinates, move_coordinates, pawn_en_passant_status, 0);
		} // end of if piece we moved was a white pawn
		else if (square[best_piece_file][best_piece_rank] == -1) // if piece we moved was a black pawn
		{
			BlackEnPassant(square, piece_coordinates, move_coordinates, pawn_en_passant_status, 0);
		} // end of if piece we moved was a black pawn

		PerformValidatedMoves(board_state, square, piece_coordinates, move_coordinates, piece_positions, 0);

		if (square[move_coordinates[0]][move_coordinates[1]] == 1 && move_coordinates[1] == 7) // if white pawn deserves promotion
		{
			square[move_coordinates[0]][move_coordinates[1]] = best_promotion;
		} // end of if white pawn deserves promotion
		else if (square[move_coordinates[0]][move_coordinates[1]] == -1 && move_coordinates[1] == 0) // black pawn deserves promotion
		{
			square[move_coordinates[0]][move_coordinates[1]] = best_promotion;
		} // end of if black pawn deserves promotion

		DisallowCastling(board_state, square);

		return 1;
	}
} // end of ComputerMoves function

/* This function performs Minimax algorithm */
int Minimax(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int **color, int ****piece_positions, unsigned int **piece_count, int *piece_value, int **pawn_en_passant_status, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int *piece_coordinates, int *move_coordinates, int ***board_history, int *alpha, int *beta)
{
	int i, j, piece_file = -9, piece_rank = -9, move_found = 0;
	int old_piece_square = -9, old_move_square = -9;
	int best_score = -999999999, best_piece_file = -9, best_piece_rank = -9, best_move_file = -9, best_move_rank = -9, best_promotion = -9;

	/* Create a structure to store the current board state before we change the working one */
	struct BoardState old_board_state;
	old_board_state = (*board_state);

	/* Create a structure to store the best board state found from searching the game tree */
	struct BoardState best_board_state;
	best_board_state = (*board_state);

	for (i = depth; i < max_depth; i++)
	{
		alpha[i] = alpha[depth - 1];
		beta[i] = beta[depth - 1];
	} // end of i loop

	if ((use_alpha_beta_pruning == 1 && (depth % 2 == 0 && best_score < beta[depth]) || (depth % 2 == 1 && -best_score > alpha[depth])) || use_alpha_beta_pruning == 0)
	{
		for (i = 0; i < chess_constants.number_of_unique_pieces; i++)
		{
			for (j = 0; j < piece_count[1 - ((turn + depth) % 2)][i]; j++)
			{
				piece_file = piece_positions[1 - ((turn + depth) % 2)][i][j][0];
				piece_rank = piece_positions[1 - ((turn + depth) % 2)][i][j][1];

				piece_coordinates[0] = piece_file;
				piece_coordinates[1] = piece_rank;

				old_piece_square = square[piece_file][piece_rank];

				if (optimized_computer_moves == 1) // if using optimized computer moves
				{
					if (old_piece_square == 1) // if piece is a white pawn
					{
						MinimaxWhitePawn(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
					} // end of if piece is a white pawn
					else if (old_piece_square == -1) // if piece is a black pawn
					{
						MinimaxBlackPawn(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
					} // end of if piece is a pawn
					else if (abs(old_piece_square) == 2) // if piece is a knight
					{
						MinimaxKnight(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
					} // end of if piece is a knight
					else if (abs(old_piece_square) == 3) // if piece is a bishop
					{
						MinimaxBishop(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
					} // end of if piece is a bishop
					else if (abs(old_piece_square) == 4) // if piece is a rook
					{
						MinimaxRook(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
					} // end of if piece is a rook
					else if (abs(old_piece_square) == 5) // if piece is a queen
					{
						MinimaxQueen(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
					} // end of if piece is a queen
					else if (abs(old_piece_square) == 6) // if piece is a king
					{
						MinimaxKing(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
					} // end of if piece is a king
				} // end of if using optimized computer moves
				else // if NOT using optimized computer moves
				{
					MinimaxUnoptimized(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, &best_score, &best_piece_file, &best_piece_rank, &best_move_file, &best_move_rank, &best_promotion, &best_board_state, alpha, beta);
				} // end of if NOT using optimized computer moves
			} // end of j loop
		} // end of i loop
	}

	return best_score;
} // end of Minimax function

/* This function applies minimax to white pawns */
void MinimaxWhitePawn(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta)
{
	unsigned int i, move_file, move_rank;
	int move_found;

	for (i = 0; i < chess_constants.number_of_possible_pawn_moves; i++)
	{
		move_found = ComputerWhitePawnMoves(piece_coordinates, move_coordinates, i);

		if (move_found == 1) // if move was found
		{
			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);
		} // end of if move was found
	} // end of i loop
} // end of MinimaxWhitePawn function

/* This function applies minimax to black pawns */
void MinimaxBlackPawn(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta)
{
	unsigned int i, move_file, move_rank;
	int move_found;

	for (i = 0; i < chess_constants.number_of_possible_pawn_moves; i++)
	{
		move_found = ComputerBlackPawnMoves(piece_coordinates, move_coordinates, i);

		if (move_found == 1) // if move was found
		{
			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);
		} // end of if move was found
	} // end of i loop
} // end of MinimaxBlackPawn function

/* This function applies minimax to knights */
void MinimaxKnight(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta)
{
	unsigned int i, move_file, move_rank;
	int move_found;

	for (i = 0; i < chess_constants.number_of_possible_knight_moves; i++)
	{
		move_found = ComputerKnightMoves(piece_coordinates, move_coordinates, i);

		if (move_found == 1) // if move was found
		{
			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);
		} // end of if move was found
	} // end of i loop
} // end of MinimaxKnight function

/* This function applies minimax to bishops */
void MinimaxBishop(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta)
{
	int i, j, move_file, move_rank;

	/* Move right and up */
	for (i = piece_coordinates[0] + 1; i < 8; i++)
	{
		j = piece_coordinates[1] + (i - piece_coordinates[0]);
		if (j <=  7)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

	/* Move left and up */
	for (i = piece_coordinates[0] - 1; i >= 0; i--)
	{
		j = piece_coordinates[1] - (i - piece_coordinates[0]);
		if (j <=  7)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

	/* Move left and down */
	for (i = piece_coordinates[0] - 1; i >= 0; i--)
	{
		j = piece_coordinates[1] + (i - piece_coordinates[0]);
		if (j >=  0)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

	/* Move right and down */
	for (i = piece_coordinates[0] + 1; i < 8; i++)
	{
		j = piece_coordinates[1] - (i - piece_coordinates[0]);
		if (j >=  0)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop
} // end of MinimaxBishop function

/* This function applies minimax to rooks */
void MinimaxRook(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta)
{
	int i, j, move_file, move_rank;

	/* Move right */
	for (i = piece_coordinates[0] + 1; i < 8; i++)
	{
		move_coordinates[0] = i;
		move_coordinates[1] = piece_coordinates[1];

		move_file = move_coordinates[0];
		move_rank = move_coordinates[1];

		AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

		if (square[i][piece_coordinates[1]] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop

	/* Move up */
	for (i = piece_coordinates[1] + 1; i < 8; i++)
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = i;

		move_file = move_coordinates[0];
		move_rank = move_coordinates[1];

		AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

		if (square[piece_coordinates[0]][i] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop

	/* Move left */
	for (i = piece_coordinates[0] - 1; i >= 0; i--)
	{
		move_coordinates[0] = i;
		move_coordinates[1] = piece_coordinates[1];

		move_file = move_coordinates[0];
		move_rank = move_coordinates[1];

		AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

		if (square[i][piece_coordinates[1]] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop

	/* Move down */
	for (i = piece_coordinates[1] - 1; i >= 0; i--)
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = i;

		move_file = move_coordinates[0];
		move_rank = move_coordinates[1];

		AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

		if (square[piece_coordinates[0]][i] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop
} // end of MinimaxRook function

/* This function applies minimax to queens */
void MinimaxQueen(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta)
{
	int i, j, move_file, move_rank;

	/* Move right and up */
	for (i = piece_coordinates[0] + 1; i < 8; i++)
	{
		j = piece_coordinates[1] + (i - piece_coordinates[0]);
		if (j <=  7)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

	/* Move left and up */
	for (i = piece_coordinates[0] - 1; i >= 0; i--)
	{
		j = piece_coordinates[1] - (i - piece_coordinates[0]);
		if (j <=  7)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

	/* Move left and down */
	for (i = piece_coordinates[0] - 1; i >= 0; i--)
	{
		j = piece_coordinates[1] + (i - piece_coordinates[0]);
		if (j >=  0)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

	/* Move right and down */
	for (i = piece_coordinates[0] + 1; i < 8; i++)
	{
		j = piece_coordinates[1] - (i - piece_coordinates[0]);
		if (j >=  0)
		{
			move_coordinates[0] = i;
			move_coordinates[1] = j;

			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

			if (square[i][j] != 0) // if square is occupied
			{
				break;
			} // end of if square is occupied
		}
		else
		{
			break;
		}
	} // end of i loop

	/* Move right */
	for (i = piece_coordinates[0] + 1; i < 8; i++)
	{
		move_coordinates[0] = i;
		move_coordinates[1] = piece_coordinates[1];

		move_file = move_coordinates[0];
		move_rank = move_coordinates[1];

		AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

		if (square[i][piece_coordinates[1]] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop

	/* Move up */
	for (i = piece_coordinates[1] + 1; i < 8; i++)
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = i;

		move_file = move_coordinates[0];
		move_rank = move_coordinates[1];

		AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

		if (square[piece_coordinates[0]][i] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop

	/* Move left */
	for (i = piece_coordinates[0] - 1; i >= 0; i--)
	{
		move_coordinates[0] = i;
		move_coordinates[1] = piece_coordinates[1];

		move_file = move_coordinates[0];
		move_rank = move_coordinates[1];

		AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

		if (square[i][piece_coordinates[1]] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop

	/* Move down */
	for (i = piece_coordinates[1] - 1; i >= 0; i--)
	{
		move_coordinates[0] = piece_coordinates[0];
		move_coordinates[1] = i;

		move_file = move_coordinates[0];
		move_rank = move_coordinates[1];

		AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

		if (square[piece_coordinates[0]][i] != 0) // if square is occupied
		{
			break;
		} // end of if square is occupied
	} // end of i loop
} // end of MinimaxQueen function

/* This function applies minimax to kings */
void MinimaxKing(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta)
{
	unsigned int i, move_file, move_rank;
	int move_found;

	for (i = 0; i < chess_constants.number_of_possible_king_moves; i++)
	{
		move_found = ComputerKingMoves(turn, depth, piece_coordinates, move_coordinates, i);

		if (move_found == 1) // if move was found
		{
			move_file = move_coordinates[0];
			move_rank = move_coordinates[1];

			AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);
		} // end of if move was found
	} // end of i loop
} // end of MinimaxKing function

/* This function applies minimax unoptimized */
void MinimaxUnoptimized(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta)
{
	unsigned int i, j, move_file, move_rank;

	for (i = 0; i < chess_constants.number_of_columns; i++)
	{
		for (j = 0; j < chess_constants.number_of_rows; j++)
		{
			if (!(i == piece_file && j == piece_rank)) // if not trying to move to same spot
			{
				move_file = i;
				move_rank = j;

				move_coordinates[0] = move_file;
				move_coordinates[1] = move_rank;

				AttemptComputerMoves(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, turn, depth, max_depth, square, piece_coordinates, move_coordinates, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, pawn_en_passant_status, piece_positions, color, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);
			} // end of if not trying to move to same spot
		} // end of j loop
	} // end of i loop
} // end of MinimaxUnoptimized function

/* This function attempts computer moves */
void AttemptComputerMoves(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int turn, int depth, int max_depth, int **square, int *piece_coordinates, int *move_coordinates, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int move_file, int move_rank, int **pawn_en_passant_status, int ****piece_positions, int **color, unsigned int **piece_count, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta)
{
	int i, j, invalid_move = 1, old_move_square = -9, promotion = -9, temp_score = -987654321;

	if (alpha[depth] < beta[depth])
	{
		invalid_move = CheckLegalComputerMove(board_state, turn, depth, square, piece_coordinates, move_coordinates, pawn_en_passant_status, piece_positions);

		//---------------------------------------------------------------------Perform Moves---------------------------------------------------------------------

		if (invalid_move == 0) // if move is valid
		{
			ApplyComputerMove(board_state, depth, piece_file, piece_rank, move_file, move_rank, &old_move_square, square, piece_coordinates, move_coordinates, pawn_en_passant_status, piece_positions);

			if ((turn + depth) % 2 == 1) // if it is white's turn
			{
				if (old_piece_square == 1 && move_rank == 7) // if white pawn deserves promotion
				{
					for (i = 2; i <= 5; i++)
					{
						/* Gets changed in Minimax function in AfterApplyingComputerMove function */
						piece_coordinates[0] = piece_file;
						piece_coordinates[1] = piece_rank;
						move_coordinates[0] = move_file;
						move_coordinates[1] = move_rank;

						promotion = i;

						square[move_file][move_rank] = promotion;

						temp_score = AfterApplyingComputerMove(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, depth, max_depth, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, old_move_square, square, color, piece_coordinates, move_coordinates, piece_count, piece_positions, pawn_en_passant_status, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, promotion, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

						if (depth == 0)
						{
							printf("\t\t\tAttemptComputerMoves: temp_score = %d, promotion = %d\n", temp_score, promotion);
						}
					} // end of i loop
				} // end of if white pawn deserves promotion
				else // if white pawn does NOT deserve promotion
				{
					temp_score = AfterApplyingComputerMove(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, depth, max_depth, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, old_move_square, square, color, piece_coordinates, move_coordinates, piece_count, piece_positions, pawn_en_passant_status, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, -9, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

					if (depth == 0)
					{
						printf("\t\t\tAttemptComputerMoves: temp_score = %d\n", temp_score);
					}
				} // end of if white pawn does NOT deserve promotion
			} // end of if it is white's turn
			else // if it is black's turn
			{
				if (old_piece_square == -1 && move_rank == 0) // if black pawn deserves promotion
				{
					for (i = 2; i <= 5; i++)
					{
						/* Gets changed in Minimax function in AfterApplyingComputerMove function */
						piece_coordinates[0] = piece_file;
						piece_coordinates[1] = piece_rank;
						move_coordinates[0] = move_file;
						move_coordinates[1] = move_rank;

						promotion = -i;

						square[move_file][move_rank] = promotion;

						temp_score = AfterApplyingComputerMove(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, depth, max_depth, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, old_move_square, square, color, piece_coordinates, move_coordinates, piece_count, piece_positions, pawn_en_passant_status, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, promotion, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

						if (depth == 0)
						{
							printf("\t\t\tAttemptComputerMoves: temp_score = %d, promotion = %d\n", temp_score, promotion);
						}
					} // end of i loop
				} // end of if black pawn deserves promotion
				else // if white black does NOT deserve promotion
				{
					temp_score = AfterApplyingComputerMove(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, depth, max_depth, piece_file, piece_rank, old_piece_square, old_board_state, move_file, move_rank, old_move_square, square, color, piece_coordinates, move_coordinates, piece_count, piece_positions, pawn_en_passant_status, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, board_history, -9, best_score, best_piece_file, best_piece_rank, best_move_file, best_move_rank, best_promotion, best_board_state, alpha, beta);

					if (depth == 0)
					{
						printf("\t\t\tAttemptComputerMoves: temp_score = %d\n", temp_score);
					}
				} // end of if black pawn does NOT deserve promotion
			} // end of if it is black's turn

			/* Gets changed in Minimax function in AfterApplyingComputerMove function */
			piece_coordinates[0] = piece_file;
			piece_coordinates[1] = piece_rank;
			move_coordinates[0] = move_file;
			move_coordinates[1] = move_rank;

			//---------------------------------------------------------------Undo move---------------------------------------------------------------
			UndoComputerMove(board_state, depth, square, piece_coordinates, move_coordinates, old_piece_square, old_move_square, pawn_en_passant_status, old_board_state);

			CountPieces(board_state, square, piece_count, piece_positions);
		} // end of if move is valid
	}
} // end of AttemptComputerMoves function

/* This function checks if computer move is legal */
int CheckLegalComputerMove(struct BoardState *board_state, int turn, int depth, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int ****piece_positions)
{
	int invalid_move = 1;

	board_state->white_queen_side_castle_allow = 0;
	board_state->white_king_side_castle_allow = 0;
	board_state->black_queen_side_castle_allow = 0;
	board_state->black_king_side_castle_allow = 0;
	board_state->en_passant_captured = -9;

	invalid_move = MovementValidation(board_state, square, piece_coordinates, move_coordinates, pawn_en_passant_status, piece_positions, 0);

	if (invalid_move == 1) // if move is still invalid
	{
		if ((turn + depth) % 2 == 1) // if it is white's turn
		{
			if (board_state->white_in_check == 0) // if white is NOT in check
			{
				WhiteQueenSideCastling(board_state, piece_coordinates, move_coordinates, square, &invalid_move);

				if (invalid_move == 1) // if move is still invalid
				{
					WhiteKingSideCastling(board_state, piece_coordinates, move_coordinates, square, &invalid_move);
				} // end of if move is still invalid
			} // end of if white is NOT in check
		} // end of if it is white's turn
		else // if it is black's turn
		{
			if (board_state->black_in_check == 0) // if black is NOT in check
			{
				BlackQueenSideCastling(board_state, piece_coordinates, move_coordinates, square, &invalid_move);

				if (invalid_move == 1) // if move is still invalid
				{
					BlackKingSideCastling(board_state, piece_coordinates, move_coordinates, square, &invalid_move);
				} // end of if move is still invalid
			} // end of if black is NOT in check
		} // end of if it is black's turn
	} // end of if move is still invalid

	return invalid_move;
} // end of CheckLegalComputerMove function

/* This function applies computer moves */
void ApplyComputerMove(struct BoardState *board_state, int depth, int piece_file, int piece_rank, int move_file, int move_rank, int *old_move_square, int **square, int *piece_coordinates, int *move_coordinates, int **pawn_en_passant_status, int ****piece_positions)
{
	if (depth == 0)
	{
		if ((board_state->turn + depth) % 2 == 1) // if it is white's turn
		{
			printf("ApplyComputerMove: Depth = %d, VALID White piece at [%c%d] with square = %d moving to [%c%d] with square = %d", depth, column_names[piece_file], piece_rank + 1, square[piece_file][piece_rank], column_names[move_file], move_rank + 1, square[move_file][move_rank]);
		} // end of if it is white's turn
		else // if it is black's turn
		{
			printf("ApplyComputerMove: Depth = %d, VALID Black piece at [%c%d] with square = %d moving to [%c%d] with square = %d", depth, column_names[piece_file], piece_rank + 1, square[piece_file][piece_rank], column_names[move_file], move_rank + 1, square[move_file][move_rank]);
		} // end of if it is black's turn
	}

	(*old_move_square) = square[move_file][move_rank];

	if (square[piece_coordinates[0]][piece_coordinates[1]] == 1) // if piece we moved was a white pawn
	{
		WhiteEnPassant(square, piece_coordinates, move_coordinates, pawn_en_passant_status, 0);
	} // end of if piece we moved was a white pawn
	else if (square[piece_coordinates[0]][piece_coordinates[1]] == -1) // if piece we moved was a black pawn
	{
		BlackEnPassant(square, piece_coordinates, move_coordinates, pawn_en_passant_status, 0);
	} // end of if piece we moved was a black pawn

	PerformValidatedMoves(board_state, square, piece_coordinates, move_coordinates, piece_positions, 0);

	DisallowCastling(board_state, square);
} // end of ApplyComputerMove function

/* This function performs remaining operations after applying computer moves */
int AfterApplyingComputerMove(FILE *outfile_computer_move_log, FILE *outfile_computer_move_score_log, FILE *outfile_piece_move_board_history, struct BoardState *board_state, int *depth_daughters, int *depth_valid_move_count, int **piece_board_history, int **move_board_history, int depth, int max_depth, int piece_file, int piece_rank, int old_piece_square, struct BoardState old_board_state, int move_file, int move_rank, int old_move_square, int **square, int **color, int *piece_coordinates, int *move_coordinates, unsigned int **piece_count, int ****piece_positions, int **pawn_en_passant_status, int *piece_value, int **non_passed_pawn_opening_values, int **non_passed_pawn_end_game_values, double **pawn_advancement_multipliers, double **knight_position_modifiers, int ***board_history, int promotion, int *best_score, int *best_piece_file, int *best_piece_rank, int *best_move_file, int *best_move_rank, int *best_promotion, struct BoardState *best_board_state, int *alpha, int *beta)
{
	int i, temp_score = -999999999, game_over = 0;

	CountPieces(board_state, square, piece_count, piece_positions);

	if ((board_state->turn + depth) % 2 == 1) // if it is white's turn
	{
		board_state->white_in_check = 0;
		board_state->black_in_check = InCheckChecker(square, piece_positions[1][5][0], -1);
	} // end of if it is white's turn
	else // if it is black's turn
	{
		board_state->black_in_check = 0;
		board_state->white_in_check = InCheckChecker(square, piece_positions[0][5][0], 1);
	} // end of if it is black's turn

	depth_valid_move_count[depth]++;

	if (activate_computer_move_log == 1)
	{
		fprintf(outfile_computer_move_log, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t", board_state->turn, depth, piece_file, piece_rank, old_piece_square, move_file, move_rank, old_move_square);
		fprintf(outfile_computer_move_log, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d", board_state->white_queen_side_castle, board_state->white_king_side_castle, board_state->black_queen_side_castle, board_state->black_king_side_castle, board_state->white_queen_side_castle_allow, board_state->white_king_side_castle_allow, board_state->black_queen_side_castle_allow, board_state->black_king_side_castle_allow, board_state->en_passant_captured, pawn_en_passant_status[0][move_coordinates[0]], pawn_en_passant_status[1][move_coordinates[0]], promotion);

		fprintf(outfile_computer_move_log, "\t%d", depth_daughters[depth]);
		for (i = 0; i < maximum_depth; i++)
		{
			fprintf(outfile_computer_move_log, "\t%d", depth_valid_move_count[i]);
		} // end of i loop
		fprintf(outfile_computer_move_log, "\n");
	}

	piece_board_history[depth][0] = piece_file;
	piece_board_history[depth][1] = piece_rank;
	piece_board_history[depth][2] = old_piece_square;

	move_board_history[depth][0] = move_file;
	move_board_history[depth][1] = move_rank;
	move_board_history[depth][2] = old_move_square;

	/******************************************************************************************************/
	/********************************* CHECK FOR CHECKMATE OR STALEMATE ***********************************/
	/******************************************************************************************************/

	game_over = NoLegalMovesMateChecker(board_state, board_state->turn + depth, square, pawn_en_passant_status, piece_count, piece_positions, color, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, 0);

	if (game_over == 1) // if there is a mate
	{
		if ((board_state->turn + depth) % 2 == 1) // if it is white's turn
		{
			/* There is a mate during white's turn */
			if (board_state->black_in_check == 1) // if black is in check
			{
				/* White checkmates black! White has won the game! */
				if (delay_wins == 1) // if we do NOT care to speed up a win
				{
					temp_score = 99999999;
				} // end of if we do NOT care to speed up a win
				else // if we want to speed up a win
				{
					temp_score = 99999999 - depth;
				} // end of if we want to speed up a win
			} // end of if black is in check
			else // if black is NOT in check
			{
				/* No valid moves! Stalemate! It is a draw! */
				if (disfavor_draws == 1) // if we want to avoid draws
				{
					if (delay_draws == 1) // if we do NOT care to speed up a draw
					{
						temp_score = -100000;
					} // end of if we do NOT care to speed up a draw
					else // if we want to speed up draws
					{
						temp_score = -100000 - depth;
					} // end of if we want to speed up draws
				} // end of if we want to avoid draws
				else // if we do NOT care to avoid draws
				{
					if (delay_draws == 1) // if we do NOT care to speed up a draw
					{
						temp_score = 0;
					} // end of if we do NOT care to speed up a draw
					else // if we want to speed up draws
					{
						temp_score = -depth;
					} // end of if we want to speed up draws
				} // end of if we do NOT care to avoid draws
			} // end of if black is NOT in check
		} // end of if it is white's turn
		else // if it is black's turn
		{
			/* There is a mate during black's turn */
			if (board_state->white_in_check == 1) // if white is in check
			{
				/* Black checkmates white! Black has won the game! */
				if (delay_wins == 1) // if we do NOT care to speed up a win
				{
					temp_score = 99999999;
				} // end of if we do NOT care to speed up a win
				else // if we want to speed up a win
				{
					temp_score = 99999999 - depth;
				} // end of if we want to speed up a win
			} // end of if white is in check
			else // if white is NOT in check
			{
				/* No valid moves! Stalemate! It is a draw! */
				if (disfavor_draws == 1) // if we want to avoid draws
				{
					if (delay_draws == 1) // if we do NOT care to speed up a draw
					{
						temp_score = -100000;
					} // end of if we do NOT care to speed up a draw
					else // if we want to speed up draws
					{
						temp_score = -100000 - depth;
					} // end of if we want to speed up draws
				} // end of if we want to avoid draws
				else // if we do NOT care to avoid draws
				{
					if (delay_draws == 1) // if we do NOT care to speed up a draw
					{
						temp_score = 0;
					} // end of if we do NOT care to speed up a draw
					else // if we want to speed up draws
					{
						temp_score = -depth;
					} // end of if we want to speed up draws
				} // end of if we do NOT care to avoid draws
			} // end of if white is NOT in check
		} // end of if it is black's turn
	} // end of if there is a mate
	else // if there is NOT a mate
	{
		/* There is NO mate. Game still on! */

		/******************************************************************************************************/
		/********************************** CHECK FOR THREEFOLD REPETITION ************************************/
		/******************************************************************************************************/

		game_over = ThreeFoldRepetition(board_state->turn, square, board_history);
		if (game_over == 1) // if there has been three fold repetition
		{
			/* The same board state has happened three times! It is a draw! */
			if (disfavor_draws == 1) // if we want to avoid draws
			{
				if (delay_draws == 1) // if we do NOT care to speed up a draw
				{
					temp_score = -100000;
				} // end of if we do NOT care to speed up a draw
				else // if we want to speed up draws
				{
					temp_score = -100000 - depth;
				} // end of if we want to speed up draws
			} // end of if we want to avoid draws
			else // if we do NOT care to avoid draws
			{
				if (delay_draws == 1) // if we do NOT care to speed up a draw
				{
					temp_score = 0;
				} // end of if we do NOT care to speed up a draw
				else // if we want to speed up draws
				{
					temp_score = -depth;
				} // end of if we want to speed up draws
			} // end of if we do NOT care to avoid draws
		} // end of if there has been three fold repetition
		else // if game did NOT end in a draw
		{
			/******************************************************************************************************/
			/************************************ CHECK FOR FIFTY MOVE RULE ***************************************/
			/******************************************************************************************************/

			game_over = FiftyMoveRule(board_state, move_coordinates, square);
			if (game_over == 1) // if there have been fifty moves without a pawn move AND piece capture
			{
				/* There hasn't been a pawn move AND piece capture in 50 turns! It is a draw! */
				if (disfavor_draws == 1) // if we want to avoid draws
				{
					if (delay_draws == 1) // if we do NOT care to speed up a draw
					{
						temp_score = -100000;
					} // end of if we do NOT care to speed up a draw
					else // if we want to speed up draws
					{
						temp_score = -100000 - depth;
					} // end of if we want to speed up draws
				} // end of if we want to avoid draws
				else // if we do NOT care to avoid draws
				{
					if (delay_draws == 1) // if we do NOT care to speed up a draw
					{
						temp_score = 0;
					} // end of if we do NOT care to speed up a draw
					else // if we want to speed up draws
					{
						temp_score = -depth;
					} // end of if we want to speed up draws
				} // end of if we do NOT care to avoid draws
			} // end of if there have been fifty moves without a pawn move AND piece capture
			else // if game did NOT end in a draw
			{
				/******************************************************************************************************/
				/******************************** CHECK FOR CHECKMATE IMPOSSIBILITY ***********************************/
				/******************************************************************************************************/

				game_over = CheckmateImpossibility(piece_count, square, color);
				if (game_over == 1) // if checkmate is impossible
				{
					/* Checkmate is impossible! It is a draw! */
					if (disfavor_draws == 1) // if we want to avoid draws
					{
						if (delay_draws == 1) // if we do NOT care to speed up a draw
						{
							temp_score = -100000;
						} // end of if we do NOT care to speed up a draw
						else // if we want to speed up draws
						{
							temp_score = -100000 - depth;
						} // end of if we want to speed up draws
					} // end of if we want to avoid draws
					else // if we do NOT care to avoid draws
					{
						if (delay_draws == 1) // if we do NOT care to speed up a draw
						{
							temp_score = 0;
						} // end of if we do NOT care to speed up a draw
						else // if we want to speed up draws
						{
							temp_score = -depth;
						} // end of if we want to speed up draws
					} // end of if we do NOT care to avoid draws
				} // end of if checkmate is impossible
				else // if game did NOT end in a draw
				{
					/******************************************************************************************************/
					/********************************* CALCULATE BOARD EVALUATION SCORE ***********************************/
					/******************************************************************************************************/

					if (depth < max_depth - 1)
					{
						temp_score = -Minimax(outfile_computer_move_log, outfile_computer_move_score_log, outfile_piece_move_board_history, board_state, depth_daughters, depth_valid_move_count, piece_board_history, move_board_history, board_state->turn, depth + 1, max_depth, square, color, piece_positions, piece_count, piece_value, pawn_en_passant_status, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers, piece_coordinates, move_coordinates, board_history, alpha, beta);
					}
					else
					{
						if ((board_state->turn + depth) % 2 == 1) // if it is white's turn
						{
							temp_score = BoardEvaluationScore(board_state, board_state->turn, square, color, piece_positions, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers);
						} // end of if it is white's turn
						else // if it is black's turn
						{
							temp_score = -BoardEvaluationScore(board_state, board_state->turn, square, color, piece_positions, piece_count, piece_value, non_passed_pawn_opening_values, non_passed_pawn_end_game_values, pawn_advancement_multipliers, knight_position_modifiers);
						} // end of if it is black's turn
					}
				} // end of if game did NOT end in a draw
			} // end of if game did NOT end in a draw
		} // end of if game did NOT end in a draw
	} // end of if there is NOT a mate

	if (activate_computer_move_score_log == 1)
	{
		fprintf(outfile_computer_move_score_log, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t", board_state->turn, depth, piece_file, piece_rank, old_piece_square, move_file, move_rank, old_move_square, temp_score);
		fprintf(outfile_computer_move_score_log, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d", board_state->white_queen_side_castle, board_state->white_king_side_castle, board_state->black_queen_side_castle, board_state->black_king_side_castle, board_state->white_queen_side_castle_allow, board_state->white_king_side_castle_allow, board_state->black_queen_side_castle_allow, board_state->black_king_side_castle_allow, board_state->en_passant_captured, pawn_en_passant_status[0][move_coordinates[0]], pawn_en_passant_status[1][move_coordinates[0]], promotion);
		fprintf(outfile_computer_move_log, "\t%d", depth_daughters[depth]);
		for (i = 0; i < maximum_depth; i++)
		{
			fprintf(outfile_computer_move_score_log, "\t%d", depth_valid_move_count[i]);
		} // end of i loop
		fprintf(outfile_computer_move_score_log, "\n");
	}

	if (activate_board_history_log == 1)
	{
		fprintf(outfile_piece_move_board_history, "%d\t%d\t%d", board_state->turn, depth, temp_score);
		for (i = 0; i <= depth; i++)
		{
			fprintf(outfile_piece_move_board_history, "\t%d\t%d\t%d\t%d\t%d\t%d", piece_board_history[i][0], piece_board_history[i][1], piece_board_history[i][2], move_board_history[i][0], move_board_history[i][1], move_board_history[i][2]);
		} // end of i loop

		for (i = depth + 1; i < maximum_depth; i++)
		{
			fprintf(outfile_piece_move_board_history, "\t%d\t%d\t%d\t%d\t%d\t%d", -9,-9,-9,-9,-9,-9);
		} // end of i loop
		fprintf(outfile_piece_move_board_history, "\n");
	}

	if (temp_score > (*best_score)) // if this moves score is better than current best score
	{
		(*best_score) = temp_score;
		(*best_piece_file) = piece_file;
		(*best_piece_rank) = piece_rank;
		(*best_move_file) = move_file;
		(*best_move_rank) = move_rank;
		(*best_promotion) = promotion;

		(*best_board_state) = (*board_state);
	} // end of if this moves score is better than current best score

	if (use_alpha_beta_pruning == 1) // if we are using alpha beta pruning
	{
		if (depth == max_depth - 1) // if at a leaf node
		{
			if (depth % 2 == 0) // if this is a maximizer
			{
				if (temp_score >= alpha[depth])
				{
					alpha[depth] = temp_score;
				}
			} // end of if this is a maximizer
			else // if this is a minimizer
			{
				if (-temp_score <= beta[depth])
				{
					beta[depth] = -temp_score;
				}
			} // end of if this is a minimizer
		} // end of if at a leaf node
		else // if not at a leaf node
		{
			if (game_over == 1) // if at a terminal node
			{
				if (depth % 2 == 0) // if this is a maximizer
				{
					if (temp_score >= alpha[depth])
					{
						alpha[depth] = temp_score;
					}
				} // end of if this is a maximizer
				else // if this is a minimizer
				{
					if (-temp_score <= beta[depth])
					{
						beta[depth] = -temp_score;
					}
				} // end of if this is a minimizer
			} // end of if at a terminal node
			else // if going back up the tree
			{
				if (depth % 2 == 0) // if this is a maximizer
				{
					if (alpha[depth] < beta[depth + 1])
					{
						alpha[depth] = beta[depth + 1];
					}
				} // end of if this is a maximizer
				else // if this is a minimizer
				{
					if (beta[depth] > alpha[depth + 1])
					{
						beta[depth] = alpha[depth + 1];
					}
				} // end of if this is a minimizer
			} // end of if going back up the tree
		} // end of if not at a leaf node
	} // end of if we are using alpha beta pruning

	return temp_score;
} // end of AfterApplyingComputerMove function

/* This function undos computer moves */
void UndoComputerMove(struct BoardState *board_state, int depth, int **square, int *piece_coordinates, int *move_coordinates, int old_piece_square, int old_move_square, int **pawn_en_passant_status, struct BoardState old_board_state)
{
	square[piece_coordinates[0]][piece_coordinates[1]] = old_piece_square;
	square[move_coordinates[0]][move_coordinates[1]] = old_move_square;

	/* Add captured pawn from en passant */
	if (old_piece_square == 1) // if piece moved was white pawn
	{
		pawn_en_passant_status[0][move_coordinates[0]] = 0;
		if (board_state->en_passant_captured != -9)
		{
			square[move_coordinates[0]][move_coordinates[1] - 1] = -1;
			board_state->en_passant_captured = -9;
		}
	} // end of if piece moved was white pawn
	else if (old_piece_square == -1) // if piece moved was black pawn
	{
		pawn_en_passant_status[1][move_coordinates[0]] = 0;
		if (board_state->en_passant_captured != -9)
		{
			square[move_coordinates[0]][move_coordinates[1] + 1] = 1;
			board_state->en_passant_captured = -9;
		}
	} // end of if piece moved was a black pawn

	/* If castled, move rook back too */
	if (board_state->white_queen_side_castle == 0 && old_board_state.white_queen_side_castle == 1)
	{
		if (board_state->white_queen_side_castle_allow == 1)
		{
			/* White UNcastled queen side! */
			square[3][0] = 0;
			square[0][0] = 4;
			board_state->white_queen_side_castle_allow = 0;
		}
		board_state->white_queen_side_castle = old_board_state.white_queen_side_castle;
	}

	if (board_state->white_king_side_castle == 0 && old_board_state.white_king_side_castle == 1)
	{
		if (board_state->white_king_side_castle_allow == 1)
		{
			/* White UNcastled king side! */
			square[5][0] = 0;
			square[7][0] = 4;
			board_state->white_king_side_castle_allow = 0;
		}
		board_state->white_king_side_castle = old_board_state.white_king_side_castle;
	}

	if (board_state->black_queen_side_castle == 0 && old_board_state.black_queen_side_castle == 1)
	{
		if (board_state->black_queen_side_castle_allow == 1)
		{
			/* Black UNcastled queen side! */
			square[3][7] = 0;
			square[0][7] = -4;
			board_state->black_queen_side_castle_allow = 0;
		}
		board_state->black_queen_side_castle = old_board_state.black_queen_side_castle;
	}

	if (board_state->black_king_side_castle == 0 && old_board_state.black_king_side_castle == 1)
	{
		if (board_state->black_king_side_castle_allow == 1)
		{
			/* Black UNcastled king side! */
			square[5][7] = 0;
			square[7][7] = -4;
			board_state->black_king_side_castle_allow = 0;
		}
		board_state->black_king_side_castle = old_board_state.black_king_side_castle;
	}

	/* Fix for 50 move rule */
	if (old_piece_square == 1 || old_piece_square == -1) // if white or black pawn moved then undo update pawn_move
	{
		board_state->pawn_move = old_board_state.pawn_move;
	} // end of if white pawn moved then undo update pawn_move

	if (board_state->white_pieces_old != old_board_state.white_pieces_old) // if the number of white pieces has changed
	{
		board_state->capture_move = old_board_state.capture_move;
		board_state->white_pieces_old = old_board_state.white_pieces_old;
	} // end of if the number of black pieces has changed

	if (board_state->black_pieces_old != old_board_state.black_pieces_old) // if the number of black pieces has changed
	{
		board_state->capture_move = old_board_state.capture_move;
		board_state->black_pieces_old = old_board_state.black_pieces_old;
	} // end of if the number of black pieces has changed
} // end of //UndoComputerMove function

/* This function tries computer white pawn moves */
int ComputerWhitePawnMoves(int *piece_coordinates, int *move_coordinates, int move_number)
{
	int move_found = 0;

	if (move_number == 0)
	{
		/* Move 2 up */
		if (piece_coordinates[1] == 1) // if white pawn in white pawn starting row
		{
			move_coordinates[0] = piece_coordinates[0];
			move_coordinates[1] = piece_coordinates[1] + 2;
			move_found = 1;
		} // end of if white pawn in white pawn starting row
	}
	else if (move_number == 1)
	{
		/* Move 1 up */
		if (piece_coordinates[1] + 1 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0];
			move_coordinates[1] = piece_coordinates[1] + 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 2)
	{
		/* Move 1 left and 1 up */
		if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] + 1 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] - 1;
			move_coordinates[1] = piece_coordinates[1] + 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 3)
	{
		/* Move 1 right and 1 up */
		if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] + 1 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] + 1;
			move_coordinates[1] = piece_coordinates[1] + 1;
			move_found = 1;
		} // end of if still on board
	}

	return move_found;
} // end of ComputerWhitePawnMoves function

/* This function tries computer black pawn moves */
int ComputerBlackPawnMoves(int *piece_coordinates, int *move_coordinates, int move_number)
{
	int move_found = 0;

	if (move_number == 0)
	{
		/* Move 2 down */
		if (piece_coordinates[1] == 6) // if black pawn in black pawn starting row
		{
			move_coordinates[0] = piece_coordinates[0];
			move_coordinates[1] = piece_coordinates[1] - 2;
			move_found = 1;
		} // end of if black pawn in black pawn starting row
	}
	else if (move_number == 1)
	{
		/* Move 1 down */
		if (piece_coordinates[1] - 1 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0];
			move_coordinates[1] = piece_coordinates[1] - 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 2)
	{
		/* Move 1 left and 1 down */
		if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] - 1 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] - 1;
			move_coordinates[1] = piece_coordinates[1] - 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 3)
	{
		/* Move 1 right and 1 down */
		if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] - 1 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] + 1;
			move_coordinates[1] = piece_coordinates[1] - 1;
			move_found = 1;
		} // end of if still on board
	}

	return move_found;
} // end of ComputerBlackPawnMoves function

/* This function tries computer knight moves */
int ComputerKnightMoves(int *piece_coordinates, int *move_coordinates, int move_number)
{
	int move_found = 0;

	if (move_number == 0)
	{
		/* Move 2 right, 1 up */
		if (piece_coordinates[0] + 2 <= 7 && piece_coordinates[1] + 1 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] + 2;
			move_coordinates[1] = piece_coordinates[1] + 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 1)
	{
		/* Move 2 right, 1 down */
		if (piece_coordinates[0] + 2 <= 7 && piece_coordinates[1] - 1 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] + 2;
			move_coordinates[1] = piece_coordinates[1] - 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 2)
	{
		/* Move 2 left, 1 up */
		if (piece_coordinates[0] - 2 >= 0 && piece_coordinates[1] + 1 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] - 2;
			move_coordinates[1] = piece_coordinates[1] + 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 3)
	{
		/* Move 2 left, 1 down */
		if (piece_coordinates[0] - 2 >= 0 && piece_coordinates[1] - 1 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] - 2;
			move_coordinates[1] = piece_coordinates[1] - 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 4)
	{
		/* Move 1 right, 2 up */
		if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] + 2 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] + 1;
			move_coordinates[1] = piece_coordinates[1] + 2;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 5)
	{
		/* Move 1 right, 2 down */
		if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] - 2 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] + 1;
			move_coordinates[1] = piece_coordinates[1] - 2;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 6)
	{
		/* Move 1 left, 2 up */
		if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] + 2 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] - 1;
			move_coordinates[1] = piece_coordinates[1] + 2;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 7)
	{
		/* Move 1 left, 2 down */
		if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] - 2 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] - 1;
			move_coordinates[1] = piece_coordinates[1] - 2;
			move_found = 1;
		} // end of if still on board
	}

	return move_found;
} // end of ComputerKnightMoves function

/* This function tries computer king moves */
int ComputerKingMoves(int turn, int depth, int *piece_coordinates, int *move_coordinates, int move_number)
{
	int move_found = 0;

	if (move_number == 0)
	{
		/* Move 1 right */
		if (piece_coordinates[0] + 1 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] + 1;
			move_coordinates[1] = piece_coordinates[1];
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 1)
	{
		/* Move 1 right, 1 up */
		if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] + 1 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] + 1;
			move_coordinates[1] = piece_coordinates[1] + 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 2)
	{
		/* Move 1 up */
		if (piece_coordinates[1] + 1 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0];
			move_coordinates[1] = piece_coordinates[1] + 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 3)
	{
		/* Move 1 left, 1 up */
		if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] + 1 <= 7) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] - 1;
			move_coordinates[1] = piece_coordinates[1] + 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 4)
	{
		/* Move 1 left */
		if (piece_coordinates[0] - 1 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] - 1;
			move_coordinates[1] = piece_coordinates[1];
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 5)
	{
		/* Move 1 left, 1 down */
		if (piece_coordinates[0] - 1 >= 0 && piece_coordinates[1] - 1 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] - 1;
			move_coordinates[1] = piece_coordinates[1] - 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 6)
	{
		/* Move 1 down */
		if (piece_coordinates[1] - 1 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0];
			move_coordinates[1] = piece_coordinates[1] - 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 7)
	{
		/* Move 1 right, 1 down */
		if (piece_coordinates[0] + 1 <= 7 && piece_coordinates[1] - 1 >= 0) // if still on board
		{
			move_coordinates[0] = piece_coordinates[0] + 1;
			move_coordinates[1] = piece_coordinates[1] - 1;
			move_found = 1;
		} // end of if still on board
	}
	else if (move_number == 8)
	{
		/* Move to queen side castle */
		if ((turn + depth) % 2 == 1) // if it is white's turn
		{
			if (piece_coordinates[0] == 4 && piece_coordinates[1] == 0) // if white king in starting square
			{
				move_coordinates[0] = 2;
				move_coordinates[1] = 0;
				move_found = 1;
			} // end of if white king in starting square
		} // end of if it is white's turn
		else // if it is black's turn
		{
			if (piece_coordinates[0] == 4 && piece_coordinates[1] == 7) // if black king in starting square
			{
				move_coordinates[0] = 2;
				move_coordinates[1] = 7;
				move_found = 1;
			} // end of if black king in starting square
		} // end of if it is black's turn
	}
	else if (move_number == 9)
	{
		/* Move to king side castle */
		if ((turn + depth) % 2 == 1) // if it is white's turn
		{
			if (piece_coordinates[0] == 4 && piece_coordinates[1] == 0) // if white king in starting square
			{
				move_coordinates[0] = 6;
				move_coordinates[1] = 0;
				move_found = 1;
			} // end of if white king in starting square
		} // end of if it is white's turn
		else // if it is black's turn
		{
			if (piece_coordinates[0] == 4 && piece_coordinates[1] == 7) // if black king in starting square
			{
				move_coordinates[0] = 6;
				move_coordinates[1] = 7;
				move_found = 1;
			} // end of if black king in starting square
		} // end of if it is black's turn
	}

	return move_found;
} // end of ComputerKnightMoves function
