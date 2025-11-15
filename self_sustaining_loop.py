"""
SELF-SUSTAINING NEURAL NETWORK TRAINING LOOP

Start with behavior cloning, then continuously improve via:
1. Play against Stockfish Level N (bullet speed)
2. Stockfish Level 8 acts as "sensei" - labels good/bad moves
3. Retrain on new data
4. When 90%+ win rate → move to next level
5. Repeat forever!

This runs WHILE data extraction continues!
"""

import modal
import pickle
from pathlib import Path

app = modal.App("knight0-self-sustaining")

# Use GPU for training!
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "python-chess>=1.999",
        "numpy>=1.24.0",
        "tqdm>=4.65.0"
    )
    .apt_install("stockfish")
    .add_local_dir("knight0", remote_path="/root/knight0_pkg/knight0")
)

volume = modal.Volume.from_name("knight0-volume", create_if_missing=True)
VOLUME_PATH = "/root/knight0"


@app.function(
    image=image,
    gpu="T4",  # GPU for fast training!
    timeout=7200,  # 2 hours
    volumes={VOLUME_PATH: volume},
)
def self_sustaining_loop():
    """
    Self-sustaining training loop that runs forever!
    """
    import sys
    sys.path.insert(0, "/root/knight0_pkg")
    
    import torch
    import chess
    import chess.engine
    from knight0.model import ChessNet, CONFIGS
    from knight0.encoding import board_to_tensor, move_to_index
    from knight0.utils import normalize_score
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import random
    
    print("="*80)
    print("🚀 SELF-SUSTAINING NEURAL NETWORK TRAINING LOOP")
    print("="*80)
    print("")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Step 1: Load or create initial model
    model_path = Path(VOLUME_PATH) / "current_model.pth"
    behavior_data_path = Path(VOLUME_PATH) / "behavior_clone_data.pkl"
    
    if model_path.exists():
        print("✓ Loading existing model...")
        model = ChessNet(**CONFIGS["small"]).to(device)
        model.load_state_dict(torch.load(model_path))
    else:
        print("✓ Creating new model...")
        model = ChessNet(**CONFIGS["small"]).to(device)
        
        # Train on behavior cloning data if available
        if behavior_data_path.exists():
            print("✓ Training on behavior cloning data...")
            with open(behavior_data_path, 'rb') as f:
                bc_data = pickle.load(f)
            
            # Quick training on behavior cloning
            quick_train(model, bc_data, device, epochs=3)
            
            # Save
            torch.save(model.state_dict(), model_path)
            volume.commit()
    
    # Step 2: Start curriculum loop
    current_level = 1  # Stockfish skill level
    sensei_level = 8   # Sensei for evaluation
    
    game_history = []  # Store all games for retraining
    
    print("")
    print("="*80)
    print("🎯 STARTING CURRICULUM TRAINING")
    print("="*80)
    print(f"Opponent: Stockfish Level {current_level}")
    print(f"Sensei: Stockfish Level {sensei_level}")
    print("")
    
    # Open engines
    with chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish") as opponent_engine, \
         chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish") as sensei_engine:
        
        # Configure engines
        opponent_engine.configure({"Skill Level": current_level})
        sensei_engine.configure({"Skill Level": sensei_level})
        
        cycle = 1
        
        while current_level <= 10:  # Go up to Level 10
            print(f"\n{'='*80}")
            print(f"CYCLE {cycle}: Training vs Level {current_level}")
            print(f"{'='*80}\n")
            
            # Play 100 bullet games (more data!)
            wins = 0
            losses = 0
            draws = 0
            games_played = 0
            new_positions = []
            
            for game_num in range(100):
                result, game_moves = play_bullet_game(
                    model=model,
                    opponent_engine=opponent_engine,
                    device=device
                )
                
                # POST-GAME ANALYSIS by sensei (much better!)
                positions = analyze_game_with_sensei(
                    game_moves=game_moves,
                    sensei_engine=sensei_engine,
                    result=result
                )
                
                if result == "win":
                    wins += 1
                    emoji = "🎉"
                elif result == "loss":
                    losses += 1
                    emoji = "💔"
                else:
                    draws += 1
                    emoji = "🤝"
                
                games_played += 1
                new_positions.extend(positions)
                game_history.extend(positions)
                
                # Calculate estimated Elo (based on Stockfish levels)
                # Stockfish Level 1 ≈ 800 Elo, Level 10 ≈ 2000 Elo
                opponent_elo = 800 + (current_level - 1) * 133  # ~133 Elo per level
                win_rate = wins / games_played if games_played > 0 else 0
                
                # Elo estimate using Elo rating formula
                # Expected score formula: E = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
                # Reverse: player_elo = opponent_elo - 400 * log10(1/win_rate - 1)
                if win_rate > 0.01 and win_rate < 0.99:
                    import math
                    estimated_elo = opponent_elo - 400 * math.log10(1/win_rate - 1)
                elif win_rate >= 0.99:
                    estimated_elo = opponent_elo + 400
                else:
                    estimated_elo = opponent_elo - 400
                
                print(f"  {emoji} Game {games_played}: {result.upper()} | "
                      f"W:{wins} D:{draws} L:{losses} ({win_rate*100:.1f}%) | "
                      f"Est. Elo: {estimated_elo:.0f} (vs {opponent_elo})")
                
                if (game_num + 1) % 10 == 0:
                    print(f"  📊 Checkpoint: {wins}/{games_played} wins, "
                          f"win rate: {win_rate*100:.1f}%, "
                          f"estimated Elo: {estimated_elo:.0f}")
            
            # Calculate win rate and final Elo
            win_rate = wins / games_played
            opponent_elo = 800 + (current_level - 1) * 133
            
            if win_rate > 0.01 and win_rate < 0.99:
                import math
                estimated_elo = opponent_elo - 400 * math.log10(1/win_rate - 1)
            elif win_rate >= 0.99:
                estimated_elo = opponent_elo + 400
            else:
                estimated_elo = opponent_elo - 400
            
            print(f"\n{'='*80}")
            print(f"📊 CYCLE {cycle} RESULTS")
            print(f"{'='*80}")
            print(f"  Games played: {games_played}")
            print(f"  Wins:   {wins} 🎉")
            print(f"  Draws:  {draws} 🤝")
            print(f"  Losses: {losses} 💔")
            print(f"  Win rate: {win_rate*100:.1f}%")
            print(f"  Opponent: Stockfish Level {current_level} (~{opponent_elo} Elo)")
            print(f"  🏆 ESTIMATED ELO: {estimated_elo:.0f}")
            print(f"  New positions: {len(new_positions):,}")
            print(f"  Total positions: {len(game_history):,}")
            print(f"{'='*80}\n")
            
            # Retrain on new data with MORE epochs for better learning!
            print(f"🔄 Retraining on {min(len(game_history), 20000):,} recent positions...")
            quick_train(model, game_history[-20000:], device, epochs=10)  # 10 epochs, last 20k positions!
            
            # Save model
            torch.save(model.state_dict(), model_path)
            volume.commit()
            print(f"✓ Model saved to volume\n")
            
            # Check if we can advance
            if win_rate >= 0.90:
                print(f"🎉🎉🎉 LEVEL UP! 🎉🎉🎉")
                print(f"Moving from Level {current_level} ({opponent_elo} Elo) to Level {current_level + 1} ({opponent_elo + 133} Elo)")
                print(f"")
                current_level += 1
                opponent_engine.configure({"Skill Level": current_level})
            else:
                print(f"⏳ Need 90%+ win rate to advance (currently {win_rate*100:.1f}%)")
                print(f"   Keep training at Level {current_level}...\n")
            
            cycle += 1
    
    print("\n" + "="*80)
    print("✓ CURRICULUM COMPLETE! Reached Level 10!")
    print("="*80)
    
    return {
        "final_level": current_level,
        "total_positions": len(game_history),
        "cycles": cycle
    }


def analyze_game_with_sensei(game_moves, sensei_engine, result):
    """
    POST-GAME ANALYSIS: Sensei reviews entire game and creates training data.
    This is MUCH better than real-time labeling!
    
    For each position where model played:
    - Sensei suggests best move
    - If model won, also include model's moves (they worked!)
    - If model lost, ONLY use sensei's suggestions
    """
    import chess
    import chess.engine
    from knight0.utils import normalize_score
    
    positions = []
    board = chess.Board()
    
    for move_info in game_moves:
        fen = move_info["fen"]
        model_move = move_info["move"]
        was_model = move_info["was_model"]
        
        if not was_model:
            # Opponent move, skip
            board.push(chess.Move.from_uci(model_move))
            continue
        
        # Sensei analyzes this position
        board_copy = chess.Board(fen)
        try:
            sensei_eval = sensei_engine.analyse(board_copy, chess.engine.Limit(nodes=1000))  # Deeper analysis!
            
            if "pv" in sensei_eval and sensei_eval["pv"]:
                sensei_best = sensei_eval["pv"][0].uci()
                
                # Get score
                score = 0
                if "score" in sensei_eval:
                    score_obj = sensei_eval["score"].relative
                    if score_obj.is_mate():
                        score = 10000 if score_obj.mate() > 0 else -10000
                    elif score_obj.score() is not None:
                        score = score_obj.score()
                
                # ALWAYS learn from sensei's best move
                positions.append({
                    "board": fen,
                    "move": sensei_best,
                    "value": normalize_score(score)
                })
                
                # If we WON and model's move was different, also learn from our move
                # (it worked, so there might be multiple good moves)
                if result == "win" and model_move != sensei_best:
                    positions.append({
                        "board": fen,
                        "move": model_move,
                        "value": normalize_score(score * 0.8)  # Slightly lower confidence
                    })
        
        except Exception as e:
            pass  # Skip positions that fail
        
        board.push(chess.Move.from_uci(model_move))
    
    return positions


def play_bullet_game(model, opponent_engine, device):
    """
    Play one bullet game against opponent.
    Returns game record for post-game analysis by sensei.
    """
    import torch
    import chess
    import chess.engine
    from knight0.encoding import board_to_tensor, move_to_index, index_to_move
    import random
    
    board = chess.Board()
    game_moves = []  # Record all moves for post-game analysis
    
    # Randomly choose color
    model_is_white = random.choice([True, False])
    
    move_num = 0
    
    while not board.is_game_over() and move_num < 100:
        move_num += 1
        
        if (board.turn == chess.WHITE) == model_is_white:
            # Model's turn - record position before move
            fen_before = board.fen()
            
            with torch.no_grad():
                board_np = board_to_tensor(board)
                board_tensor = torch.from_numpy(board_np).unsqueeze(0).float().to(device)
                policy_logits, value = model(board_tensor)
                policy_probs = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
            
            # Get legal moves
            legal_moves = list(board.legal_moves)
            legal_indices = [move_to_index(m) for m in legal_moves]
            
            # Sample from policy with exploration
            legal_probs = [policy_probs[idx] for idx in legal_indices]
            total_prob = sum(legal_probs)
            if total_prob > 0:
                legal_probs = [p / total_prob for p in legal_probs]
                # Add epsilon-greedy exploration
                if random.random() < 0.1:  # 10% random moves for exploration
                    chosen_idx = random.randint(0, len(legal_moves) - 1)
                else:
                    chosen_idx = random.choices(range(len(legal_moves)), weights=legal_probs)[0]
                move = legal_moves[chosen_idx]
            else:
                move = random.choice(legal_moves)
            
            # Record this move for post-game analysis
            game_moves.append({
                "fen": fen_before,
                "move": move.uci(),
                "was_model": True
            })
            
            board.push(move)
        
        else:
            # Opponent's turn (bullet speed!)
            fen_before = board.fen()
            opp_move = opponent_engine.play(board, chess.engine.Limit(time=0.05))
            
            # Record opponent move too
            game_moves.append({
                "fen": fen_before,
                "move": opp_move.move.uci(),
                "was_model": False
            })
            
            board.push(opp_move.move)
    
    # Determine result
    outcome = board.outcome()
    if outcome:
        if outcome.winner == chess.WHITE:
            game_result = "win" if model_is_white else "loss"
        elif outcome.winner == chess.BLACK:
            game_result = "loss" if model_is_white else "win"
        else:
            game_result = "draw"
    else:
        game_result = "draw"
    
    return game_result, game_moves


def quick_train(model, positions, device, epochs=10):
    """Improved training with better hyperparameters."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from knight0.encoding import board_to_tensor, move_to_index
    import chess
    
    if len(positions) == 0:
        print("    ⚠️  No positions to train on!")
        return
    
    class SimpleDataset(Dataset):
        def __init__(self, positions):
            self.positions = positions
        
        def __len__(self):
            return len(self.positions)
        
        def __getitem__(self, idx):
            pos = self.positions[idx]
            board = chess.Board(pos["board"])
            board_np = board_to_tensor(board)
            board_tensor = torch.from_numpy(board_np).float()
            move_idx = move_to_index(chess.Move.from_uci(pos["move"]))
            value = pos["value"]
            return board_tensor, move_idx, value
    
    dataset = SimpleDataset(positions)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)  # Larger batch
    
    # Better optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    model.train()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        batches = 0
        
        for boards, moves, values in loader:
            boards = boards.to(device)
            moves = moves.to(device)
            values = values.to(device).float()
            
            optimizer.zero_grad()
            
            policy_logits, value_pred = model(boards)
            
            policy_loss = policy_criterion(policy_logits, moves)
            value_loss = value_criterion(value_pred.squeeze(), values)
            
            # Weighted loss (policy more important for gameplay)
            loss = 2.0 * policy_loss + value_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batches += 1
        
        scheduler.step()
        
        avg_loss = total_loss / batches
        avg_policy = total_policy_loss / batches
        avg_value = total_value_loss / batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            indicator = "✓"
        else:
            indicator = " "
        
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"    {indicator} Epoch {epoch+1}/{epochs}: "
                  f"Loss={avg_loss:.4f} (Policy={avg_policy:.4f}, Value={avg_value:.4f}) "
                  f"LR={scheduler.get_last_lr()[0]:.2e}")


@app.local_entrypoint()
def main():
    """Launch the self-sustaining loop!"""
    print("🚀 Launching Self-Sustaining Training Loop...")
    print("")
    print("This will:")
    print("  1. Train on behavior cloning data")
    print("  2. Play vs Stockfish Level 1")
    print("  3. Learn from Level 8 'sensei'")
    print("  4. Retrain on new games")
    print("  5. Level up when 90%+ win rate")
    print("  6. Repeat until Level 10!")
    print("")
    
    result = self_sustaining_loop.remote()
    
    print("\n✓ Self-sustaining loop complete!")
    print(f"  Final level: {result['final_level']}")
    print(f"  Total positions: {result['total_positions']:,}")
    print(f"  Training cycles: {result['cycles']}")

