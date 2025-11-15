"""
Data source management for downloading and organizing chess game datasets.

Supports:
- Lichess Elite Database
- Lumbra's Gigabase (OTB)
- TCEC engine games
- CCRL engine games
- Hugging Face chess games dataset
"""

import os
import requests
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DataSourceManager:
    """
    Manager for downloading and organizing chess game datasets.
    """
    
    def __init__(self, root_dir: str):
        """
        Args:
            root_dir: Root directory for storing data (e.g., /root/knight0)
        """
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different sources
        self.lichess_dir = self.raw_dir / "lichess"
        self.lumbra_dir = self.raw_dir / "lumbra"
        self.tcec_dir = self.raw_dir / "tcec"
        self.ccrl_dir = self.raw_dir / "ccrl"
        
        for d in [self.lichess_dir, self.lumbra_dir, self.tcec_dir, self.ccrl_dir]:
            d.mkdir(exist_ok=True)
    
    def download_file(self, url: str, output_path: Path, chunk_size: int = 8192) -> bool:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            output_path: Local path to save to
            chunk_size: Download chunk size in bytes
            
        Returns:
            True if successful, False otherwise
        """
        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            return True
        
        try:
            logger.info(f"Downloading {url} to {output_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_lichess_elite(self, year: Optional[int] = None) -> List[Path]:
        """
        Download Lichess Elite Database games.
        
        Note: The actual URLs need to be determined from https://database.nikonoel.fr/
        This is a placeholder implementation.
        
        Args:
            year: Optional year to download (e.g., 2023)
            
        Returns:
            List of downloaded file paths
        """
        logger.info("Lichess Elite Database download not fully implemented.")
        logger.info("Please manually download PGN files from https://database.nikonoel.fr/")
        logger.info(f"Save them to: {self.lichess_dir}")
        
        # TODO: Implement actual download logic once URLs are known
        # Example structure:
        # urls = [
        #     "https://database.nikonoel.fr/lichess_elite_2023_01.pgn.zst",
        #     ...
        # ]
        # for url in urls:
        #     filename = Path(url).name
        #     self.download_file(url, self.lichess_dir / filename)
        
        return list(self.lichess_dir.glob("*.pgn*"))
    
    def download_lumbra_gigabase(self) -> List[Path]:
        """
        Download Lumbra's Gigabase OTB games.
        
        Note: This requires manual download from the website.
        
        Returns:
            List of downloaded file paths
        """
        logger.info("Lumbra's Gigabase download requires manual download.")
        logger.info("Please download PGN files from https://lumbrasgigabase.com/en/download-in-pgn-format-en/")
        logger.info(f"Save them to: {self.lumbra_dir}")
        
        return list(self.lumbra_dir.glob("*.pgn*"))
    
    def download_tcec_games(self, season: Optional[int] = None) -> List[Path]:
        """
        Download TCEC engine games.
        
        Args:
            season: Optional season number to download
            
        Returns:
            List of downloaded file paths
        """
        logger.info("TCEC games can be downloaded from GitHub releases:")
        logger.info("https://github.com/TCEC-Chess/tcecgames/releases")
        logger.info(f"Save them to: {self.tcec_dir}")
        
        # TODO: Implement actual download from GitHub releases
        # Example:
        # urls = [
        #     "https://github.com/TCEC-Chess/tcecgames/releases/download/S25/S25_premier.pgn.gz",
        #     ...
        # ]
        
        return list(self.tcec_dir.glob("*.pgn*"))
    
    def download_ccrl_games(self) -> List[Path]:
        """
        Download CCRL engine games.
        
        Returns:
            List of downloaded file paths
        """
        logger.info("CCRL games (without comments) can be downloaded from:")
        logger.info("https://www.computerchess.org.uk/ccrl/404/games.html")
        logger.info(f"Save them to: {self.ccrl_dir}")
        
        # TODO: Implement actual download
        # Example URL (this may change):
        # url = "https://www.computerchess.org.uk/ccrl/404/games.pgn.gz"
        
        return list(self.ccrl_dir.glob("*.pgn*"))
    
    def download_hf_dataset(self, cache_dir: Optional[str] = None) -> str:
        """
        Load the Hugging Face chess games dataset.
        
        This uses the datasets library to load angeluriot/chess_games.
        
        Args:
            cache_dir: Optional cache directory for HF datasets
            
        Returns:
            Dataset identifier (for use with datasets.load_dataset)
        """
        logger.info("To use the Hugging Face dataset:")
        logger.info("  from datasets import load_dataset")
        logger.info("  dataset = load_dataset('angeluriot/chess_games')")
        logger.info("This will be handled in extract_positions.py")
        
        return "angeluriot/chess_games"
    
    def list_available_pgns(self) -> List[Path]:
        """
        List all available PGN files in the raw directory.
        
        Returns:
            List of PGN file paths
        """
        pgn_files = []
        for pattern in ["*.pgn", "*.pgn.gz", "*.pgn.bz2", "*.pgn.zst"]:
            pgn_files.extend(self.raw_dir.rglob(pattern))
        
        logger.info(f"Found {len(pgn_files)} PGN files")
        for f in pgn_files:
            logger.info(f"  - {f.relative_to(self.raw_dir)}")
        
        return pgn_files
    
    def setup_quick_test_data(self) -> Path:
        """
        Create a small test PGN file for quick development/testing.
        
        Returns:
            Path to test PGN file
        """
        test_pgn = self.raw_dir / "test_games.pgn"
        
        if test_pgn.exists():
            logger.info(f"Test PGN already exists: {test_pgn}")
            return test_pgn
        
        # A few high-quality games for testing
        test_games = """
[Event "World Championship"]
[Site "Moscow"]
[Date "1985.09.03"]
[Round "1"]
[White "Kasparov, Garry"]
[Black "Karpov, Anatoly"]
[Result "1-0"]
[WhiteElo "2700"]
[BlackElo "2720"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6
8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8
14. Ng3 g6 15. a4 c5 16. d5 c4 17. Bg5 h6 18. Be3 Nc5 19. Qd2 h5
20. Nh2 Qc7 21. Ng4 hxg4 22. hxg4 Nh7 23. Qf2 Be7 24. Qh4 Bg5 25. Qh5 Bxe3
26. fxe3 Qd7 27. Rf1 Nf6 28. Qh2 Ne8 29. Rf2 Ng7 30. Raf1 Rf8 31. Nf5 Nxf5
32. gxf5 f6 33. Rg2 Kh7 34. Rfg1 Rg8 35. g4 Bc8 36. Kf2 Bd7 37. Rh1+ Kg7
38. Be2 Be8 39. Qh6+ Kf7 40. Rhg1 Rh8 41. Qxh8 1-0

[Event "TCEC Season 24"]
[Site "Chess.com"]
[Date "2023.01.15"]
[Round "1"]
[White "Stockfish"]
[Black "Leela Chess Zero"]
[Result "1/2-1/2"]
[WhiteElo "3600"]
[BlackElo "3600"]

1. d4 Nf6 2. c4 e6 3. Nf3 d5 4. Nc3 Be7 5. Bf4 O-O 6. e3 c5 7. dxc5 Bxc5
8. a3 Nc6 9. Qc2 Qa5 10. O-O-O Be7 11. h4 dxc4 12. g4 b5 13. g5 Nd5
14. Nxd5 exd5 15. Bxc4 bxc4 16. Qxc4 Be6 17. Qc5 Qxc5 18. Bxc5 Bxc5
19. Rxd5 Rac8 20. Kb1 Na5 21. Rhd1 Bb3 22. Rd7 Bxd1 23. Rxd1 Nc4
24. Nd4 Nxa3+ 25. bxa3 Bxd4 26. Rxd4 Rc1+ 27. Kb2 Rc2+ 28. Kb3 Rxf2
29. Rd7 a5 30. Ra7 Ra8 31. Rxa8+ 1/2-1/2

[Event "Lichess Titled Arena"]
[Site "lichess.org"]
[Date "2023.06.20"]
[Round "1"]
[White "DrNykterstein"]
[Black "Chesswarrior7197"]
[Result "1-0"]
[WhiteElo "2850"]
[BlackElo "2750"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6
8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 b4 13. Ne2 Ne8
14. f4 a5 15. f5 Bc4 16. Nbd4 exd4 17. Nxd4 b3 18. Kb1 bxc2+ 19. Qxc2 Rc8
20. Bd3 Bxd3 21. Qxd3 Nc5 22. Qe2 Ne4 23. g6 fxg6 24. fxg6 h6 25. Rhf1 Rxf1
26. Rxf1 Qb6 27. Qh5 Nf6 28. Qxh6 gxh6 29. Bxh6 Qxd4 30. Rf4 1-0
"""
        
        with open(test_pgn, 'w') as f:
            f.write(test_games.strip())
        
        logger.info(f"Created test PGN: {test_pgn}")
        return test_pgn


if __name__ == "__main__":
    import tempfile
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DataSourceManager(tmpdir)
        
        print("\nCreating test data:")
        test_pgn = manager.setup_quick_test_data()
        print(f"Test PGN created at: {test_pgn}")
        
        print("\nListing available PGNs:")
        pgn_files = manager.list_available_pgns()

