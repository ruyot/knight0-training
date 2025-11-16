use chess::{Board, Color, Piece, Square, ALL_SQUARES};
use ndarray::Array4;

/// Encode chess board into 21x8x8 tensor
/// 
/// Planes:
/// 0-5: White pieces (P, N, B, R, Q, K)
/// 6-11: Black pieces (P, N, B, R, Q, K)
/// 12: Repetition count
/// 13-14: White castling rights (kingside, queenside)
/// 15-16: Black castling rights (kingside, queenside)
/// 17: Side to move (1 if white, 0 if black)
/// 18: Halfmove clock / 100
/// 19: En passant square
/// 20: Constant 1s plane
pub fn board_to_tensor(board: &Board) -> Array4<f32> {
    let mut tensor = Array4::<f32>::zeros((21, 8, 8));
    
    // Encode pieces (planes 0-11)
    for square in ALL_SQUARES {
        if let Some(piece) = board.piece_on(square) {
            let color = board.color_on(square).unwrap();
            
            let plane_offset = if color == Color::White { 0 } else { 6 };
            let piece_idx = match piece {
                Piece::Pawn => 0,
                Piece::Knight => 1,
                Piece::Bishop => 2,
                Piece::Rook => 3,
                Piece::Queen => 4,
                Piece::King => 5,
            };
            
            let plane = plane_offset + piece_idx;
            let (rank, file) = square_to_coords(square);
            tensor[[plane, rank, file]] = 1.0;
        }
    }
    
    // Castling rights (planes 13-16)
    let castles = board.castle_rights(Color::White);
    if castles.has_kingside() {
        tensor.slice_mut(ndarray::s![13, .., ..]).fill(1.0);
    }
    if castles.has_queenside() {
        tensor.slice_mut(ndarray::s![14, .., ..]).fill(1.0);
    }
    
    let castles_black = board.castle_rights(Color::Black);
    if castles_black.has_kingside() {
        tensor.slice_mut(ndarray::s![15, .., ..]).fill(1.0);
    }
    if castles_black.has_queenside() {
        tensor.slice_mut(ndarray::s![16, .., ..]).fill(1.0);
    }
    
    // Side to move (plane 17)
    if board.side_to_move() == Color::White {
        tensor.slice_mut(ndarray::s![17, .., ..]).fill(1.0);
    }
    
    // Halfmove clock (plane 18)
    let halfmove = (board.halfmove_clock() as f32) / 100.0;
    tensor.slice_mut(ndarray::s![18, .., ..]).fill(halfmove);
    
    // En passant (plane 19)
    if let Some(ep_square) = board.en_passant() {
        let (rank, file) = square_to_coords(ep_square);
        tensor[[19, rank, file]] = 1.0;
    }
    
    // Constant plane (plane 20)
    tensor.slice_mut(ndarray::s![20, .., ..]).fill(1.0);
    
    tensor
}

fn square_to_coords(square: Square) -> (usize, usize) {
    let rank = square.get_rank().to_index();
    let file = square.get_file().to_index();
    (rank, file)
}

