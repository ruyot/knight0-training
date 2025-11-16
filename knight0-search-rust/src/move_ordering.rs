use chess::{Board, ChessMove, Piece, Square};
use std::collections::HashMap;

const MAX_PLY: usize = 100;

pub struct MoveOrdering {
    // Killer moves: [ply][slot]
    killer_moves: Vec<[Option<ChessMove>; 2]>,
    
    // History heuristic: piece -> to_square -> score
    history: HashMap<(Piece, Square), i32>,
}

impl MoveOrdering {
    pub fn new() -> Self {
        Self {
            killer_moves: vec![[None; 2]; MAX_PLY],
            history: HashMap::new(),
        }
    }
    
    pub fn clear(&mut self) {
        self.killer_moves = vec![[None; 2]; MAX_PLY];
    }
    
    pub fn add_killer(&mut self, mv: ChessMove, ply: u8) {
        let ply = ply as usize;
        if ply >= MAX_PLY {
            return;
        }
        
        if Some(mv) != self.killer_moves[ply][0] {
            self.killer_moves[ply][1] = self.killer_moves[ply][0];
            self.killer_moves[ply][0] = Some(mv);
        }
    }
    
    pub fn update_history(&mut self, mv: ChessMove, depth: u8) {
        // We don't have piece info in ChessMove directly
        // This is simplified - in production you'd track piece types
        let key = (Piece::Pawn, mv.get_dest());  // Simplified
        let score = self.history.entry(key).or_insert(0);
        *score += (depth as i32) * (depth as i32);
    }
    
    pub fn order_moves(
        &self,
        board: &Board,
        moves: &[ChessMove],
        ply: u8,
        tt_move: Option<ChessMove>,
        move_probs: &HashMap<ChessMove, f32>,
    ) -> Vec<ChessMove> {
        let mut scored_moves: Vec<(i32, ChessMove)> = moves
            .iter()
            .map(|&mv| {
                let score = self.score_move(board, mv, ply, tt_move, move_probs);
                (score, mv)
            })
            .collect();
        
        scored_moves.sort_by(|a, b| b.0.cmp(&a.0));
        scored_moves.into_iter().map(|(_, mv)| mv).collect()
    }
    
    fn score_move(
        &self,
        board: &Board,
        mv: ChessMove,
        ply: u8,
        tt_move: Option<ChessMove>,
        move_probs: &HashMap<ChessMove, f32>,
    ) -> i32 {
        // 1. TT move
        if Some(mv) == tt_move {
            return 1_000_000;
        }
        
        // 2. Promotions
        if mv.get_promotion().is_some() {
            return 900_000;
        }
        
        // 3. Captures (MVV-LVA)
        if let Some(victim) = board.piece_on(mv.get_dest()) {
            let victim_value = piece_value(victim);
            let attacker = board.piece_on(mv.get_source());
            let attacker_value = attacker.map_or(0, piece_value);
            return 800_000 + victim_value * 10 - attacker_value;
        }
        
        // 4. Killer moves
        let ply = (ply as usize).min(MAX_PLY - 1);
        if Some(mv) == self.killer_moves[ply][0] {
            return 700_000;
        }
        if Some(mv) == self.killer_moves[ply][1] {
            return 600_000;
        }
        
        // 5. History heuristic (simplified)
        let history_score = 0;  // Simplified for now
        
        // 6. NN policy
        let nn_score = move_probs.get(&mv).copied().unwrap_or(0.0);
        500_000 + history_score + (nn_score * 1000.0) as i32
    }
}

fn piece_value(piece: Piece) -> i32 {
    use chess::Piece::*;
    match piece {
        Pawn => 100,
        Knight => 300,
        Bishop => 300,
        Rook => 500,
        Queen => 900,
        King => 20000,
    }
}

