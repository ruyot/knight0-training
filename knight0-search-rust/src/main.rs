use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

mod search;
mod encoding;
mod tt;
mod move_ordering;
mod eval;

use search::SearchEngine;

#[derive(Parser)]
#[command(name = "knight0-search")]
#[command(about = "High-performance chess search for knight0", long_about = None)]
struct Args {
    /// Path to ONNX model
    #[arg(short, long, default_value = "knight0_model.onnx")]
    model: PathBuf,
    
    /// FEN position to analyze
    #[arg(short, long)]
    fen: Option<String>,
    
    /// Search depth
    #[arg(short, long, default_value = "12")]
    depth: u8,
    
    /// Time limit in seconds
    #[arg(short, long, default_value = "1.0")]
    time: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("knight0 Search Engine (Rust)");
    println!("=============================\n");
    
    // Load model
    println!("Loading model from {:?}...", args.model);
    let mut engine = SearchEngine::new(&args.model)?;
    println!("Model loaded successfully\n");
    
    // Parse position
    let fen = args.fen.unwrap_or_else(|| {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()
    });
    
    println!("Position: {}", fen);
    
    // Search
    println!("\nSearching (depth={}, time={}s)...\n", args.depth, args.time);
    let result = engine.search(&fen, args.depth, args.time)?;
    
    println!("\n{}", result);
    
    Ok(())
}

