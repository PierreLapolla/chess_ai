import chess
import torch


class BoardTokenizer:
    """
    Tokenizes a chess board state into a sequence of integer tokens.

    This tokenizer converts a chess.Board object into a tensor of shape (64,)
    where each element is an integer representing a piece or an empty square.

    Token mapping:
        0 : empty square
        1 : white pawn
        2 : white knight
        3 : white bishop
        4 : white rook
        5 : white queen
        6 : white king
        7 : black pawn
        8 : black knight
        9 : black bishop
        10: black rook
        11: black queen
        12: black king
    """

    piece_to_token = {
        None: 0,
        chess.Piece(chess.PAWN, chess.WHITE): 1,
        chess.Piece(chess.KNIGHT, chess.WHITE): 2,
        chess.Piece(chess.BISHOP, chess.WHITE): 3,
        chess.Piece(chess.ROOK, chess.WHITE): 4,
        chess.Piece(chess.QUEEN, chess.WHITE): 5,
        chess.Piece(chess.KING, chess.WHITE): 6,
        chess.Piece(chess.PAWN, chess.BLACK): 7,
        chess.Piece(chess.KNIGHT, chess.BLACK): 8,
        chess.Piece(chess.BISHOP, chess.BLACK): 9,
        chess.Piece(chess.ROOK, chess.BLACK): 10,
        chess.Piece(chess.QUEEN, chess.BLACK): 11,
        chess.Piece(chess.KING, chess.BLACK): 12,
    }

    def tokenize(self, board: chess.Board) -> torch.Tensor:
        """
        Converts a chess.Board to a tensor of shape (64,) with integer tokens.
        Pre-allocates the tensor with zeros for empty squares, then fills in tokens
        using board.piece_map() for positions that have a piece.
        """
        tokens = torch.zeros(64, dtype=torch.long)
        for square, piece in board.piece_map().items():
            tokens[square] = self.piece_to_token.get(piece, 0)
        return tokens
