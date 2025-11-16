use anyhow::Result;
use chess::{Board, ChessMove};
use ndarray::{Array, Array4, ArrayView4};
use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel, tensor::OrtOwnedTensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::encoding::board_to_tensor;

pub struct NNEvaluator {
    session: Session<'static>,
    env: Arc<Environment>,
}

impl NNEvaluator {
    pub fn new(model_path: &Path) -> Result<Self> {
        let env = Arc::new(
            Environment::builder()
                .with_name("knight0")
                .with_log_level(onnxruntime::LoggingLevel::Warning)
                .build()?
        );
        
        let session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(model_path)?;
        
        Ok(Self { session, env })
    }
    
    pub fn evaluate(&self, board: &Board) -> (f32, HashMap<ChessMove, f32>) {
        // Encode board
        let board_array = board_to_tensor(board);
        
        // Run inference
        let input_tensor = board_array.into_shape((1, 21, 8, 8)).unwrap();
        
        let outputs: Vec<OrtOwnedTensor<f32, _>> = self
            .session
            .run(vec![input_tensor])
            .expect("Failed to run inference");
        
        // Extract value
        let value_output = &outputs[1];
        let value = value_output.view().iter().next().copied().unwrap_or(0.0);
        
        // Extract policy
        let policy_output = &outputs[0];
        let policy_logits = policy_output.view();
        
        // Softmax and get legal move probabilities
        let mut move_probs = HashMap::new();
        
        // TODO: Implement proper move indexing
        // For now, return empty probs (will fall back to other ordering)
        
        (value, move_probs)
    }
}

