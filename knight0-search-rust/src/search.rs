use anyhow::Result;
use chess::{Board, ChessMove, Color, MoveGen, EMPTY};
use std::path::Path;
use std::time::Instant;
use std::fmt;

use crate::tt::{TranspositionTable, TTEntry, TTFlag};
use crate::move_ordering::MoveOrdering;
use crate::eval::NNEvaluator;

const INFINITY: f32 = 999.0;
const MATE_SCORE: f32 = 10.0;

// Search parameters
const NULL_MOVE_REDUCTION: u8 = 2;
const LMR_FULL_DEPTH_MOVES: usize = 4;
const LMR_REDUCTION_LIMIT: u8 = 3;

#[derive(Debug)]
pub struct SearchResult {
    pub best_move: ChessMove,
    pub eval: f32,
    pub depth: u8,
    pub nodes: u64,
    pub time_ms: u128,
    pub pv: Vec<ChessMove>,
}

impl fmt::Display for SearchResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let pv_str: Vec<String> = self.pv.iter().map(|m| m.to_string()).collect();
        write!(
            f,
            "Best: {} (eval={:+.3}, depth={}, nodes={}, time={}ms, nps={})\nPV: {}",
            self.best_move,
            self.eval,
            self.depth,
            self.nodes,
            self.time_ms,
            if self.time_ms > 0 { (self.nodes as u128 * 1000) / self.time_ms } else { 0 },
            pv_str.join(" ")
        )
    }
}

pub struct SearchEngine {
    evaluator: NNEvaluator,
    tt: TranspositionTable,
    move_ordering: MoveOrdering,
    nodes: u64,
}

impl SearchEngine {
    pub fn new(model_path: &Path) -> Result<Self> {
        Ok(Self {
            evaluator: NNEvaluator::new(model_path)?,
            tt: TranspositionTable::new(10_000_000),
            move_ordering: MoveOrdering::new(),
            nodes: 0,
        })
    }
    
    pub fn search(&mut self, fen: &str, max_depth: u8, time_limit: f64) -> Result<SearchResult> {
        let board = Board::from_str(fen).map_err(|e| anyhow::anyhow!("Invalid FEN: {}", e))?;
        
        let start = Instant::now();
        self.nodes = 0;
        self.tt.new_search();
        self.move_ordering.clear();
        
        let mut best_move = None;
        let mut best_eval = 0.0;
        let mut best_pv = Vec::new();
        
        // Iterative deepening
        for depth in 1..=max_depth {
            if start.elapsed().as_secs_f64() > time_limit {
                break;
            }
            
            let mut pv = Vec::new();
            
            // Aspiration windows after depth 4
            let eval = if depth >= 5 {
                let window = 0.3;
                let mut alpha = best_eval - window;
                let mut beta = best_eval + window;
                
                let score = self.alpha_beta(&board, depth, alpha, beta, 0, &mut pv, true);
                
                // Re-search if failed
                if score <= alpha || score >= beta {
                    alpha = -INFINITY;
                    beta = INFINITY;
                    pv.clear();
                    self.alpha_beta(&board, depth, alpha, beta, 0, &mut pv, true)
                } else {
                    score
                }
            } else {
                self.alpha_beta(&board, depth, -INFINITY, INFINITY, 0, &mut pv, true)
            };
            
            if !pv.is_empty() {
                best_move = Some(pv[0]);
                best_eval = eval;
                best_pv = pv.clone();
            }
            
            let elapsed = start.elapsed().as_secs_f64();
            let nps = if elapsed > 0.0 { (self.nodes as f64 / elapsed) as u64 } else { 0 };
            
            let pv_str: Vec<String> = pv.iter().map(|m| m.to_string()).collect();
            println!(
                "depth {:2}: eval={:+.3}, nodes={:8}, nps={:8}, pv={}",
                depth, eval, self.nodes, nps, pv_str.join(" ")
            );
        }
        
        let time_ms = start.elapsed().as_millis();
        
        Ok(SearchResult {
            best_move: best_move.unwrap_or_else(|| {
                MoveGen::new_legal(&board).next().expect("No legal moves")
            }),
            eval: best_eval,
            depth: max_depth.saturating_sub(1),
            nodes: self.nodes,
            time_ms,
            pv: best_pv,
        })
    }
    
    fn alpha_beta(
        &mut self,
        board: &Board,
        depth: u8,
        mut alpha: f32,
        mut beta: f32,
        ply: u8,
        pv: &mut Vec<ChessMove>,
        do_null: bool,
    ) -> f32 {
        self.nodes += 1;
        
        // Draw detection
        if ply > 0 {
            // Simple repetition check (can be improved)
            if board.checkers() == &EMPTY && self.is_draw_by_material(board) {
                return 0.0;
            }
        }
        
        // Mate distance pruning
        alpha = alpha.max(-MATE_SCORE + ply as f32);
        beta = beta.min(MATE_SCORE - ply as f32);
        if alpha >= beta {
            return alpha;
        }
        
        // TT probe
        let hash = board.get_hash();
        let tt_move = if let Some(entry) = self.tt.probe(hash, depth) {
            match entry.flag {
                TTFlag::Exact => return entry.score,
                TTFlag::LowerBound => alpha = alpha.max(entry.score),
                TTFlag::UpperBound => beta = beta.min(entry.score),
            }
            if alpha >= beta {
                return entry.score;
            }
            entry.best_move
        } else {
            None
        };
        
        // Terminal node
        if depth == 0 {
            return self.quiescence(board, alpha, beta, ply, 8);
        }
        
        let in_check = board.checkers() != &EMPTY;
        
        // Check for game over
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if legal_moves.is_empty() {
            return if in_check {
                -MATE_SCORE + ply as f32
            } else {
                0.0
            };
        }
        
        // NULL MOVE PRUNING
        if do_null && !in_check && depth >= 3 && ply > 0 && !self.is_endgame(board) {
            if let Some(null_board) = board.null_move() {
                let null_score = -self.alpha_beta(
                    &null_board,
                    depth.saturating_sub(1 + NULL_MOVE_REDUCTION),
                    -beta,
                    -beta + 0.001,
                    ply + 1,
                    &mut Vec::new(),
                    false,
                );
                
                if null_score >= beta {
                    return beta;
                }
            }
        }
        
        // Evaluate for move ordering
        let (value, move_probs) = self.evaluator.evaluate(board);
        
        // Check extension
        let actual_depth = if in_check { depth + 1 } else { depth };
        
        // Order moves
        let ordered_moves = self.move_ordering.order_moves(
            board, &legal_moves, ply, tt_move, &move_probs
        );
        
        let mut best_move = None;
        let mut best_score = -INFINITY;
        let mut flag = TTFlag::UpperBound;
        let mut local_pv = Vec::new();
        
        for (move_idx, &mv) in ordered_moves.iter().enumerate() {
            let new_board = board.make_move_new(mv);
            
            // LATE MOVE REDUCTIONS
            let mut reduction = 0u8;
            if move_idx >= LMR_FULL_DEPTH_MOVES 
                && depth >= 3 
                && !in_check 
                && new_board.checkers() == &EMPTY
                && !self.is_capture(board, mv)
                && mv.get_promotion().is_none() {
                
                reduction = 1 + (move_idx / 6) as u8;
                reduction = reduction.min(LMR_REDUCTION_LIMIT);
                reduction = reduction.min(depth - 1);
            }
            
            let mut child_pv = Vec::new();
            
            let score = if reduction > 0 {
                // Search with reduction
                let score = -self.alpha_beta(
                    &new_board,
                    actual_depth.saturating_sub(1 + reduction),
                    -alpha - 0.001,
                    -alpha,
                    ply + 1,
                    &mut child_pv,
                    true,
                );
                
                // Re-search if promising
                if score > alpha {
                    child_pv.clear();
                    -self.alpha_beta(
                        &new_board,
                        actual_depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        &mut child_pv,
                        true,
                    )
                } else {
                    score
                }
            } else if move_idx == 0 {
                // PVS: first move with full window
                -self.alpha_beta(
                    &new_board,
                    actual_depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    &mut child_pv,
                    true,
                )
            } else {
                // PVS: null window search
                let score = -self.alpha_beta(
                    &new_board,
                    actual_depth - 1,
                    -alpha - 0.001,
                    -alpha,
                    ply + 1,
                    &mut Vec::new(),
                    true,
                );
                
                // Re-search if failed high
                if score > alpha && score < beta {
                    child_pv.clear();
                    -self.alpha_beta(
                        &new_board,
                        actual_depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        &mut child_pv,
                        true,
                    )
                } else {
                    score
                }
            };
            
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
                
                if score > alpha {
                    alpha = score;
                    flag = TTFlag::Exact;
                    local_pv = vec![mv];
                    local_pv.extend(child_pv);
                    
                    // Update history
                    if !self.is_capture(board, mv) && mv.get_promotion().is_none() {
                        self.move_ordering.update_history(mv, depth);
                    }
                }
                
                if score >= beta {
                    flag = TTFlag::LowerBound;
                    
                    // Killer move
                    if !self.is_capture(board, mv) && mv.get_promotion().is_none() {
                        self.move_ordering.add_killer(mv, ply);
                    }
                    
                    break;
                }
            }
        }
        
        // Store in TT
        self.tt.store(hash, depth, best_score, best_move, flag);
        
        if flag == TTFlag::Exact {
            *pv = local_pv;
        }
        
        best_score
    }
    
    fn quiescence(
        &mut self,
        board: &Board,
        mut alpha: f32,
        beta: f32,
        ply: u8,
        max_depth: u8,
    ) -> f32 {
        self.nodes += 1;
        
        let in_check = board.checkers() != &EMPTY;
        
        // Stand pat
        if !in_check {
            let (stand_pat, _) = self.evaluator.evaluate(board);
            
            if stand_pat >= beta {
                return beta;
            }
            if alpha < stand_pat {
                alpha = stand_pat;
            }
            
            if max_depth == 0 {
                return stand_pat;
            }
        } else if max_depth == 0 {
            // Extend in check
            return self.quiescence(board, alpha, beta, ply, 1);
        }
        
        // Generate tactical moves
        let moves: Vec<ChessMove> = if in_check {
            MoveGen::new_legal(board).collect()
        } else {
            MoveGen::new_legal(board)
                .filter(|&m| self.is_capture(board, m) || m.get_promotion().is_some())
                .collect()
        };
        
        if moves.is_empty() {
            return if in_check {
                -MATE_SCORE + ply as f32
            } else {
                alpha
            };
        }
        
        for &mv in &moves {
            let new_board = board.make_move_new(mv);
            let score = -self.quiescence(&new_board, -beta, -alpha, ply + 1, max_depth - 1);
            
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        
        alpha
    }
    
    fn is_capture(&self, board: &Board, mv: ChessMove) -> bool {
        board.piece_on(mv.get_dest()).is_some()
    }
    
    fn is_endgame(&self, board: &Board) -> bool {
        // Simple heuristic: count pieces
        let white_pieces = board.color_combined(Color::White).popcnt();
        let black_pieces = board.color_combined(Color::Black).popcnt();
        white_pieces <= 6 || black_pieces <= 6
    }
    
    fn is_draw_by_material(&self, board: &Board) -> bool {
        // KvK, KNvK, KBvK, KNNvK
        let white_pieces = board.color_combined(Color::White).popcnt();
        let black_pieces = board.color_combined(Color::Black).popcnt();
        
        // Very simple check
        white_pieces <= 2 && black_pieces <= 2
    }
}

