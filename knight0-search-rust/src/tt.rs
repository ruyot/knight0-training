use chess::ChessMove;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Debug, Clone)]
pub struct TTEntry {
    pub depth: u8,
    pub score: f32,
    pub best_move: Option<ChessMove>,
    pub flag: TTFlag,
    pub generation: u32,
}

pub struct TranspositionTable {
    table: HashMap<u64, TTEntry>,
    max_size: usize,
    generation: u32,
}

impl TranspositionTable {
    pub fn new(max_size: usize) -> Self {
        Self {
            table: HashMap::with_capacity(max_size.min(1_000_000)),
            max_size,
            generation: 0,
        }
    }
    
    pub fn new_search(&mut self) {
        self.generation += 1;
        
        // Clear old entries if table is getting too big
        if self.table.len() > self.max_size {
            let cutoff_gen = self.generation.saturating_sub(3);
            self.table.retain(|_, entry| entry.generation >= cutoff_gen);
        }
    }
    
    pub fn probe(&self, hash: u64, depth: u8) -> Option<TTEntry> {
        self.table.get(&hash).and_then(|entry| {
            if entry.depth >= depth {
                Some(entry.clone())
            } else {
                // Return move hint even if depth is insufficient
                Some(TTEntry {
                    depth: 0,
                    score: entry.score,
                    best_move: entry.best_move,
                    flag: entry.flag,
                    generation: entry.generation,
                })
            }
        })
    }
    
    pub fn store(
        &mut self,
        hash: u64,
        depth: u8,
        score: f32,
        best_move: Option<ChessMove>,
        flag: TTFlag,
    ) {
        let existing = self.table.get(&hash);
        
        // Replace if: new entry, deeper search, or same generation
        let should_replace = existing.map_or(true, |entry| {
            depth >= entry.depth || entry.generation < self.generation
        });
        
        if should_replace {
            self.table.insert(
                hash,
                TTEntry {
                    depth,
                    score,
                    best_move,
                    flag,
                    generation: self.generation,
                },
            );
        }
    }
}

